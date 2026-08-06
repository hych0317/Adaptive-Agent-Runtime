"""Deterministic, operator-gated Phase 4-C auto-adaptation policy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol
from uuid import UUID, NAMESPACE_URL, uuid5

from adaptive_agent_runtime import RunStatus, RuntimeEvent, StateStore, TraceSink
from adaptive_agent_runtime.decisioning import (
    DecisionCheckpointConflictError,
    DecisionReconciliationStatus,
    DecisionResultStatus,
    decision_fingerprint,
)
from adaptive_agent_runtime.optimization import (
    AutoAdaptationPolicy,
    AutoAdaptationSkipReason,
    AutoAdaptationStatus,
    AutoAdaptationTriggerRecord,
    OptimizationProposal,
    OptimizationProposalStatus,
    OptimizationRiskClassification,
    OptimizationScope,
    OptimizationTargetKey,
    RuntimeConfigurationSnapshot,
    stable_auto_adaptation_attempt_id,
    stable_auto_adaptation_trigger_identity,
    stable_optimization_apply_request_id,
)
from adaptive_agent_runtime.persistence import (
    PersistenceConflictError,
    default_runtime_configuration_snapshot,
)

from applications.research_agent.optimization_apply import (
    ResearchOptimizationConfigurationGateway,
)
from applications.research_agent.report import GovernanceRecord


class _ProposalAuthorityQuery(Protocol):
    async def list_for_scope(
        self,
        scope: OptimizationScope,
    ) -> tuple[OptimizationProposal, ...]: ...

    async def load_by_id(self, proposal_id: UUID) -> OptimizationProposal | None: ...

    async def verify_phase3_provenance(self, proposal_id: UUID) -> bool: ...


class _ConfigurationAuthorityQuery(Protocol):
    async def load_active(
        self,
        scope: OptimizationScope,
        target_key: OptimizationTargetKey,
    ) -> RuntimeConfigurationSnapshot | None: ...

    async def was_proposal_applied(self, proposal_id: UUID) -> bool: ...


class _TriggerStore(Protocol):
    async def load_for_run(
        self,
        trigger_run_id: UUID,
    ) -> AutoAdaptationTriggerRecord | None: ...

    async def claim(
        self,
        record: AutoAdaptationTriggerRecord,
    ) -> AutoAdaptationTriggerRecord: ...

    async def transition(
        self,
        record: AutoAdaptationTriggerRecord,
    ) -> AutoAdaptationTriggerRecord: ...


class ResearchAutoAdaptationCoordinator:
    """Select zero or one Proposal; all authority remains in Phase 4-B Apply."""

    module_id = "research_agent.auto_adaptation"

    def __init__(
        self,
        *,
        policy: AutoAdaptationPolicy,
        proposals: _ProposalAuthorityQuery,
        configurations: _ConfigurationAuthorityQuery,
        state_store: StateStore,
        triggers: _TriggerStore,
        gateway: ResearchOptimizationConfigurationGateway,
        trace_sink: TraceSink,
    ) -> None:
        self._policy = policy
        self._proposals = proposals
        self._configurations = configurations
        self._state_store = state_store
        self._triggers = triggers
        self._gateway = gateway
        self._trace_sink = trace_sink
        self.governance_record: GovernanceRecord | None = None

    async def evaluate_after_run(
        self,
        *,
        trigger_run_id: UUID,
        run_configuration: RuntimeConfigurationSnapshot,
    ) -> AutoAdaptationTriggerRecord:
        existing = await self._triggers.load_for_run(trigger_run_id)
        if existing is not None:
            if existing.status in {
                AutoAdaptationStatus.SELECTED,
                AutoAdaptationStatus.REVIEW_PENDING,
            }:
                if not self._policy.enabled:
                    assert existing.selected_proposal_id is not None
                    if not await self._configurations.was_proposal_applied(
                        existing.selected_proposal_id
                    ):
                        return await self._finish(
                            existing,
                            status=AutoAdaptationStatus.SKIPPED,
                            skip_reason=AutoAdaptationSkipReason.DISABLED,
                            reason=(
                                "operator disabled auto-adaptation before commit"
                            ),
                        )
                return await self._apply_selected(existing)
            return existing

        if not self._policy.enabled:
            return await self._claim_skip(
                trigger_run_id,
                run_configuration,
                AutoAdaptationSkipReason.DISABLED,
                candidate_set_fingerprint=decision_fingerprint(()),
            )
        state = await self._state_store.load(trigger_run_id)
        if state is None or state.status is not RunStatus.COMPLETED:
            return await self._claim_skip(
                trigger_run_id,
                run_configuration,
                AutoAdaptationSkipReason.RUN_NOT_COMPLETED,
                candidate_set_fingerprint=decision_fingerprint(()),
            )
        active = await self._configurations.load_active(
            self._policy.scope,
            self._policy.target_key,
        ) or default_runtime_configuration_snapshot(self._policy.scope)
        if (
            active.revision != run_configuration.revision
            or active.snapshot_fingerprint != run_configuration.snapshot_fingerprint
        ):
            return await self._claim_skip(
                trigger_run_id,
                run_configuration,
                AutoAdaptationSkipReason.BASELINE_STALE,
                candidate_set_fingerprint=decision_fingerprint(()),
                reasons=("active configuration changed while the Run was executing",),
            )

        try:
            listed = await self._proposals.list_for_scope(self._policy.scope)
        except Exception as exc:
            return await self._claim_skip(
                trigger_run_id,
                run_configuration,
                AutoAdaptationSkipReason.INCOMPLETE_PHASE3_PROVENANCE,
                candidate_set_fingerprint=decision_fingerprint(()),
                reasons=(f"proposal query failed closed: {type(exc).__name__}",),
            )
        candidate_set_fingerprint = decision_fingerprint(
            tuple(
                sorted(
                    (
                        str(item.proposal_id),
                        decision_fingerprint(item),
                    )
                    for item in listed
                )
            )
        )
        eligible: list[OptimizationProposal] = []
        rejected: list[tuple[AutoAdaptationSkipReason, str]] = []
        for listed_proposal in listed:
            proposal = await self._proposals.load_by_id(listed_proposal.proposal_id)
            if proposal is None or proposal != listed_proposal:
                rejected.append(
                    (
                        AutoAdaptationSkipReason.INCOMPLETE_PHASE3_PROVENANCE,
                        f"{listed_proposal.proposal_id}: authority readback failed",
                    )
                )
                continue
            reason = await self._ineligible_reason(proposal, active)
            if reason is None:
                eligible.append(proposal)
            else:
                rejected.append((reason, f"{proposal.proposal_id}: {reason.value}"))

        if len(eligible) != 1:
            reason = (
                AutoAdaptationSkipReason.MULTIPLE_ELIGIBLE_PROPOSALS
                if len(eligible) > 1
                else (
                    rejected[0][0]
                    if len(listed) == 1 and rejected
                    else AutoAdaptationSkipReason.NO_ELIGIBLE_PROPOSAL
                )
            )
            return await self._claim_skip(
                trigger_run_id,
                run_configuration,
                reason,
                candidate_set_fingerprint=candidate_set_fingerprint,
                reasons=tuple(item[1] for item in rejected),
            )

        proposal = eligible[0]
        trigger_identity = stable_auto_adaptation_trigger_identity(
            trigger_run_id,
            proposal.proposal_id,
            active.revision,
        )
        request_id = stable_optimization_apply_request_id(
            proposal.proposal_id,
            trigger_run_id=trigger_run_id,
            baseline_revision=active.revision,
        )
        claimed = await self._claim(
            AutoAdaptationTriggerRecord(
                attempt_id=stable_auto_adaptation_attempt_id(trigger_run_id),
                trigger_run_id=trigger_run_id,
                policy_fingerprint=self._policy.fingerprint,
                baseline_revision=active.revision,
                baseline_fingerprint=active.snapshot_fingerprint,
                candidate_set_fingerprint=candidate_set_fingerprint,
                status=AutoAdaptationStatus.SELECTED,
                selected_proposal_id=proposal.proposal_id,
                trigger_identity=trigger_identity,
                apply_request_id=request_id,
            )
        )
        if claimed.status is not AutoAdaptationStatus.SELECTED:
            return claimed
        return await self._apply_selected(claimed)

    async def contain_infrastructure_failure(
        self,
        *,
        trigger_run_id: UUID,
        run_configuration: RuntimeConfigurationSnapshot,
        error: BaseException,
    ) -> AutoAdaptationTriggerRecord:
        """Return a structured fail-closed outcome without masking the Run result."""

        summary = _error_summary(error)
        existing: AutoAdaptationTriggerRecord | None = None
        try:
            existing = await self._triggers.load_for_run(trigger_run_id)
        except Exception as ledger_error:
            summary = _merge_error_summaries(summary, _error_summary(ledger_error))

        if existing is not None:
            if existing.status in {
                AutoAdaptationStatus.SELECTED,
                AutoAdaptationStatus.REVIEW_PENDING,
            }:
                return await self._mark_unknown_after_infrastructure_failure(
                    existing,
                    summary,
                )
            return existing.model_copy(
                update={
                    "error_summary": _merge_error_summaries(
                        existing.error_summary,
                        summary,
                    )
                }
            )

        failed = create_unpersisted_auto_adaptation_failure_outcome(
            policy=self._policy,
            trigger_run_id=trigger_run_id,
            run_configuration=run_configuration,
            error=error,
        ).model_copy(
            update={
                "error_summary": summary,
                "candidate_rejection_reasons": (summary,),
                "outcome_persisted": True,
            }
        )
        try:
            claimed = await self._triggers.claim(failed)
        except Exception as ledger_error:
            return failed.model_copy(
                update={
                    "error_summary": _merge_error_summaries(
                        summary,
                        _error_summary(ledger_error),
                    ),
                    "outcome_persisted": False,
                }
            )
        if claimed.status in {
            AutoAdaptationStatus.SELECTED,
            AutoAdaptationStatus.REVIEW_PENDING,
        }:
            return await self._mark_unknown_after_infrastructure_failure(
                claimed,
                summary,
            )
        await self._best_effort_trace(claimed)
        if claimed.status is AutoAdaptationStatus.FAILED:
            return claimed
        return claimed.model_copy(
            update={
                "error_summary": _merge_error_summaries(
                    claimed.error_summary,
                    summary,
                )
            }
        )

    async def _ineligible_reason(
        self,
        proposal: OptimizationProposal,
        active: RuntimeConfigurationSnapshot,
    ) -> AutoAdaptationSkipReason | None:
        if proposal.scope != self._policy.scope:
            return AutoAdaptationSkipReason.SCOPE_MISMATCH
        if proposal.target_key is not OptimizationTargetKey.PLANNER_MAX_NODES:
            return AutoAdaptationSkipReason.TARGET_NOT_ALLOWED
        if proposal.status is not OptimizationProposalStatus.ACTIVE:
            return AutoAdaptationSkipReason.PROPOSAL_INACTIVE
        if (
            proposal.expires_at is not None
            and proposal.expires_at <= datetime.now(timezone.utc)
        ):
            return AutoAdaptationSkipReason.PROPOSAL_EXPIRED
        if (
            proposal.current_configuration_revision != active.revision
            or proposal.current_configuration_fingerprint
            != active.snapshot_fingerprint
            or proposal.current_value != active.value
            or proposal.current_value_fingerprint != active.value_fingerprint
        ):
            return AutoAdaptationSkipReason.BASELINE_STALE
        if proposal.risk_classification is not OptimizationRiskClassification.LOW:
            return AutoAdaptationSkipReason.RISK_NOT_LOW
        if proposal.counterevidence_refs:
            return AutoAdaptationSkipReason.COUNTEREVIDENCE_PRESENT
        current = active.value
        proposed = proposal.proposed_value
        if (
            isinstance(current, bool)
            or isinstance(proposed, bool)
            or not isinstance(current, int)
            or not isinstance(proposed, int)
            or not self._policy.minimum_value
            <= proposed
            <= self._policy.maximum_value
            or abs(proposed - current) != self._policy.required_delta
        ):
            return AutoAdaptationSkipReason.CHANGE_NOT_ONE
        if await self._configurations.was_proposal_applied(proposal.proposal_id):
            return AutoAdaptationSkipReason.PROPOSAL_ALREADY_APPLIED
        if not await self._proposals.verify_phase3_provenance(proposal.proposal_id):
            return AutoAdaptationSkipReason.INCOMPLETE_PHASE3_PROVENANCE
        return None

    async def _apply_selected(
        self,
        record: AutoAdaptationTriggerRecord,
    ) -> AutoAdaptationTriggerRecord:
        assert record.selected_proposal_id is not None
        assert record.trigger_identity is not None
        try:
            result = await self._gateway.request_policy_auto_apply(
                record.selected_proposal_id,
                trigger_run_id=record.trigger_run_id,
                auto_adaptation_policy_fingerprint=record.policy_fingerprint,
                auto_adaptation_trigger_id=record.trigger_identity,
                candidate_set_fingerprint=record.candidate_set_fingerprint,
                baseline_revision=record.baseline_revision,
            )
            self.governance_record = result.governance_record
        except DecisionCheckpointConflictError:
            # Another coordinator owns the same stable Decision request.  A
            # checkpoint CAS conflict says nothing about whether its Effect was
            # committed, so it must not poison the durable trigger as UNKNOWN.
            # Keep the original claim resumable; a later invocation will load
            # and reconcile the one authoritative Decision checkpoint.
            recovered = await self._triggers.load_for_run(record.trigger_run_id)
            if recovered is None:
                raise
            return recovered
        except ValueError as exc:
            return await self._finish(
                record,
                status=AutoAdaptationStatus.SKIPPED,
                skip_reason=AutoAdaptationSkipReason.BASELINE_STALE,
                reason=f"Apply validation rejected stale trigger: {exc}",
            )
        except Exception as exc:
            return await self._finish(
                record,
                status=AutoAdaptationStatus.UNKNOWN,
                skip_reason=AutoAdaptationSkipReason.RECONCILIATION_UNKNOWN,
                reason=f"Apply failed closed: {type(exc).__name__}: {exc}",
            )
        if result.receipt is not None:
            return await self._finish(
                record,
                status=AutoAdaptationStatus.APPLIED,
                apply_effect_fingerprint=result.receipt.effect_fingerprint,
                active_revision=result.receipt.active_revision,
                reason="sole eligible Proposal committed through governed Apply",
            )
        if result.review_pending:
            if record.status is AutoAdaptationStatus.REVIEW_PENDING:
                # Re-reading an unresolved Human Review is an idempotent no-op,
                # not a new trigger revision.
                return record
            return await self._finish(
                record,
                status=AutoAdaptationStatus.REVIEW_PENDING,
                skip_reason=AutoAdaptationSkipReason.REVIEW_PENDING,
                reason=result.reason or "Governance review is pending",
            )
        if result.reconciliation_status is DecisionReconciliationStatus.UNKNOWN:
            return await self._finish(
                record,
                status=AutoAdaptationStatus.UNKNOWN,
                skip_reason=AutoAdaptationSkipReason.RECONCILIATION_UNKNOWN,
                reason=result.reason or "Apply reconciliation is unknown",
            )
        if result.decision_status is DecisionResultStatus.REJECTED:
            return await self._finish(
                record,
                status=AutoAdaptationStatus.DENIED,
                skip_reason=AutoAdaptationSkipReason.GOVERNANCE_DENIED,
                reason=result.reason or "Governance denied automatic Apply",
            )
        return await self._finish(
            record,
            status=AutoAdaptationStatus.SKIPPED,
            skip_reason=AutoAdaptationSkipReason.APPLY_REJECTED,
            reason=result.reason or "Automatic Apply did not commit",
        )

    async def _claim_skip(
        self,
        trigger_run_id: UUID,
        run_configuration: RuntimeConfigurationSnapshot,
        reason: AutoAdaptationSkipReason,
        *,
        candidate_set_fingerprint: str,
        reasons: tuple[str, ...] = (),
    ) -> AutoAdaptationTriggerRecord:
        return await self._claim(
            AutoAdaptationTriggerRecord(
                attempt_id=stable_auto_adaptation_attempt_id(trigger_run_id),
                trigger_run_id=trigger_run_id,
                policy_fingerprint=self._policy.fingerprint,
                baseline_revision=run_configuration.revision,
                baseline_fingerprint=run_configuration.snapshot_fingerprint,
                candidate_set_fingerprint=candidate_set_fingerprint,
                status=AutoAdaptationStatus.SKIPPED,
                skip_reason=reason,
                candidate_rejection_reasons=reasons,
                error_summary=(
                    "; ".join(reasons)[:1024]
                    if reasons
                    else None
                ),
            )
        )

    async def _claim(
        self,
        record: AutoAdaptationTriggerRecord,
    ) -> AutoAdaptationTriggerRecord:
        try:
            claimed = await self._triggers.claim(record)
        except PersistenceConflictError:
            recovered = await self._triggers.load_for_run(record.trigger_run_id)
            if recovered is not None:
                claimed = recovered
            elif record.status is AutoAdaptationStatus.SELECTED:
                conflict_skip = record.model_copy(
                    update={
                        "status": AutoAdaptationStatus.SKIPPED,
                        "selected_proposal_id": None,
                        "trigger_identity": None,
                        "apply_request_id": None,
                        "skip_reason": (
                            AutoAdaptationSkipReason.PROPOSAL_ALREADY_APPLIED
                        ),
                        "candidate_rejection_reasons": (
                            "concurrent trigger claimed the selected Proposal",
                        ),
                    }
                )
                claimed = await self._triggers.claim(conflict_skip)
            else:
                raise
        await self._trace(claimed)
        return claimed

    async def _finish(
        self,
        record: AutoAdaptationTriggerRecord,
        *,
        status: AutoAdaptationStatus,
        reason: str,
        skip_reason: AutoAdaptationSkipReason | None = None,
        apply_effect_fingerprint: str | None = None,
        active_revision: int | None = None,
    ) -> AutoAdaptationTriggerRecord:
        finished = record.model_copy(
            update={
                "status": status,
                "skip_reason": skip_reason,
                "apply_effect_fingerprint": apply_effect_fingerprint,
                "active_revision": active_revision,
                "candidate_rejection_reasons": (
                    *record.candidate_rejection_reasons,
                    reason,
                ),
                "error_summary": (
                    reason
                    if status is not AutoAdaptationStatus.APPLIED
                    else None
                ),
                "revision": record.revision + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        stored = await self._triggers.transition(finished)
        await self._trace(stored)
        return stored

    async def _mark_unknown_after_infrastructure_failure(
        self,
        record: AutoAdaptationTriggerRecord,
        summary: str,
    ) -> AutoAdaptationTriggerRecord:
        unknown = record.model_copy(
            update={
                "status": AutoAdaptationStatus.UNKNOWN,
                "skip_reason": AutoAdaptationSkipReason.RECONCILIATION_UNKNOWN,
                "candidate_rejection_reasons": (
                    *record.candidate_rejection_reasons,
                    summary,
                ),
                "error_summary": summary,
                "revision": record.revision + 1,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        try:
            stored = await self._triggers.transition(unknown)
        except Exception as ledger_error:
            return unknown.model_copy(
                update={
                    "error_summary": _merge_error_summaries(
                        summary,
                        _error_summary(ledger_error),
                    ),
                    "outcome_persisted": False,
                }
            )
        await self._best_effort_trace(stored)
        return stored

    async def _best_effort_trace(
        self,
        record: AutoAdaptationTriggerRecord,
    ) -> None:
        try:
            await self._trace(record)
        except Exception:
            # The structured outcome is returned even when the optional audit
            # sink is unavailable.  A durable trigger remains authoritative.
            return

    async def _trace(self, record: AutoAdaptationTriggerRecord) -> None:
        await self._trace_sink.record(
            RuntimeEvent(
                event_id=uuid5(
                    NAMESPACE_URL,
                    "adaptive-agent-runtime:auto-adaptation-trace:"
                    f"{record.attempt_id}:{record.revision}:{record.status.value}",
                ),
                run_id=record.trigger_run_id,
                kind=f"optimization.auto_adaptation.{record.status.value}",
                source=self.module_id,
                occurred_at=record.updated_at,
                payload={
                    "attempt_id": str(record.attempt_id),
                    "status": record.status.value,
                    "skip_reason": (
                        record.skip_reason.value
                        if record.skip_reason is not None
                        else None
                    ),
                    "selected_proposal_id": (
                        str(record.selected_proposal_id)
                        if record.selected_proposal_id is not None
                        else None
                    ),
                    "baseline_revision": record.baseline_revision,
                    "policy_fingerprint": record.policy_fingerprint,
                    "apply_request_id": (
                        str(record.apply_request_id)
                        if record.apply_request_id is not None
                        else None
                    ),
                    "apply_effect_fingerprint": record.apply_effect_fingerprint,
                    "active_revision": record.active_revision,
                    "error_summary": record.error_summary,
                    "outcome_persisted": record.outcome_persisted,
                },
            )
        )


def _error_summary(error: BaseException) -> str:
    detail = " ".join(str(error).split())
    summary = type(error).__name__ if not detail else f"{type(error).__name__}: {detail}"
    return summary[:1024]


def _merge_error_summaries(first: str | None, second: str) -> str:
    if not first:
        return second[:1024]
    if second in first:
        return first[:1024]
    return f"{first}; {second}"[:1024]


def create_unpersisted_auto_adaptation_failure_outcome(
    *,
    policy: AutoAdaptationPolicy,
    trigger_run_id: UUID,
    run_configuration: RuntimeConfigurationSnapshot,
    error: BaseException,
) -> AutoAdaptationTriggerRecord:
    """Build the caller-visible fallback used when the control plane is down."""

    summary = _error_summary(error)
    return AutoAdaptationTriggerRecord(
        attempt_id=stable_auto_adaptation_attempt_id(trigger_run_id),
        trigger_run_id=trigger_run_id,
        policy_fingerprint=policy.fingerprint,
        baseline_revision=run_configuration.revision,
        baseline_fingerprint=run_configuration.snapshot_fingerprint,
        candidate_set_fingerprint=decision_fingerprint(()),
        status=AutoAdaptationStatus.FAILED,
        skip_reason=AutoAdaptationSkipReason.INFRASTRUCTURE_FAILURE,
        candidate_rejection_reasons=(summary,),
        error_summary=summary,
        outcome_persisted=False,
    )
