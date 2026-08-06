"""Small coordinator for one Runtime-owned decision lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from time import monotonic
from typing import Generic, TypeVar, cast
from uuid import UUID

from pydantic import JsonValue

from adaptive_agent_runtime.core.models import utc_now
from adaptive_agent_runtime.decisioning.budget import (
    budget_violations,
    can_call_agent,
    can_start_cycle,
    consume_agent_call,
    consume_decision_cycle,
)
from adaptive_agent_runtime.decisioning.context import (
    ContextProjectionPolicy,
    ProjectionSources,
)
from adaptive_agent_runtime.decisioning.contracts import (
    AgentContextBuilder,
    DecisionApplier,
    DecisionBasisProvider,
    DecisionCheckpointStore,
    DecisionGovernancePort,
    DecisionGovernanceResolution,
    DecisionProposalProducer,
    DecisionTraceWriter,
    DecisionValidator,
)
from adaptive_agent_runtime.decisioning.errors import (
    DecisionInvariantError,
    DecisionResumeError,
)
from adaptive_agent_runtime.decisioning.models import (
    AgentCallResult,
    DecisionApplyReceipt,
    DecisionCheckpoint,
    DecisionCheckpointStage,
    DecisionFaultPoint,
    DecisionGovernanceOutcome,
    DecisionGovernanceReceipt,
    DecisionRequest,
    DecisionResult,
    DecisionResultStatus,
    DecisionReconciliationStatus,
    DecisionTraceEvent,
    DecisionTraceKind,
    DecisionValidationStatus,
    DecisionValidation,
    EffectPayloadT,
    ProposalPayloadT,
    RequestPayloadT,
    ValidatedDecision,
    decision_fingerprint,
)


ApprovalT = TypeVar("ApprovalT")


class DecisionLifecycleCoordinator(
    Generic[RequestPayloadT, ProposalPayloadT, EffectPayloadT, ApprovalT]
):
    """Coordinate one decision; never schedule tasks or run an Agent loop."""

    module_id = "decision.lifecycle"

    def __init__(
        self,
        *,
        context_builder: AgentContextBuilder,
        proposal_producer: DecisionProposalProducer[ProposalPayloadT],
        basis_provider: DecisionBasisProvider,
        validator: DecisionValidator[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        governance: DecisionGovernancePort[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
            ApprovalT,
        ],
        applier: DecisionApplier[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
            ApprovalT,
        ],
        checkpoint_store: DecisionCheckpointStore[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        checkpoint_type: type[
            DecisionCheckpoint[
                RequestPayloadT,
                ProposalPayloadT,
                EffectPayloadT,
            ]
        ],
        trace_writer: DecisionTraceWriter,
        clock: Callable[[], datetime] = utc_now,
        monotonic_clock: Callable[[], float] = monotonic,
        fault_injector: Callable[
            [DecisionFaultPoint, DecisionCheckpoint[RequestPayloadT, ProposalPayloadT, EffectPayloadT]],
            None,
        ]
        | None = None,
    ) -> None:
        self._context_builder = context_builder
        self._proposal_producer = proposal_producer
        self._basis_provider = basis_provider
        self._validator = validator
        self._governance = governance
        self._applier = applier
        self._checkpoint_store = checkpoint_store
        self._checkpoint_type = checkpoint_type
        self._trace_writer = trace_writer
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._fault_injector = fault_injector

    async def run(
        self,
        request: DecisionRequest[RequestPayloadT],
        *,
        sources: ProjectionSources,
        policy: ContextProjectionPolicy,
    ) -> DecisionCheckpoint[
        RequestPayloadT,
        ProposalPayloadT,
        EffectPayloadT,
    ]:
        existing = await self._checkpoint_store.load(request.request_id)
        if existing is not None:
            persisted_request = existing.request.model_dump(mode="json")
            comparable_request = request.model_dump(mode="json")
            comparable_request["created_at"] = persisted_request["created_at"]
            if decision_fingerprint(persisted_request) != decision_fingerprint(
                comparable_request
            ):
                raise DecisionInvariantError(
                    "decision request identity was reused with another snapshot"
                )
            return await self.resume(request.request_id)
        checkpoint = self._checkpoint_type(
            request_id=request.request_id,
            run_id=request.correlation.run_id,
            stage=DecisionCheckpointStage.REQUESTED,
            request=request,
        )
        await self._checkpoint_store.save(checkpoint, expected_revision=None)
        await self._trace(checkpoint, DecisionTraceKind.REQUESTED)

        if not can_start_cycle(request.budget, checkpoint.budget_usage):
            return await self._expire(checkpoint, "decision cycle budget exhausted")
        if not can_call_agent(request.budget, checkpoint.budget_usage):
            return await self._expire(checkpoint, "Agent call budget exhausted")

        started = self._monotonic_clock()
        usage = consume_decision_cycle(checkpoint.budget_usage)
        try:
            context = self._context_builder.build(request, sources, policy)
        except Exception as exc:
            return await self._fail(checkpoint, "context projection", exc)
        usage = usage.model_copy(
            update={
                "elapsed_seconds": max(
                    usage.elapsed_seconds,
                    self._monotonic_clock() - started,
                )
            }
        )
        checkpoint = await self._advance(
            checkpoint,
            stage=DecisionCheckpointStage.CONTEXT_PROJECTED,
            budget_usage=usage,
            context_manifest=context.manifest(),
        )
        await self._trace(
            checkpoint,
            DecisionTraceKind.CONTEXT_PROJECTED,
            payload={
                "agent_scope": context.agent_scope,
                "policy_id": context.projection_policy_id,
                "context_fingerprint": context.context_fingerprint,
                "included_source_count": len(context.blocks),
                "omitted_source_count": len(context.omissions),
                "estimated_tokens": sum(
                    item.estimated_tokens for item in context.blocks
                ),
            },
        )
        exhausted = budget_violations(request.budget, usage)
        if exhausted:
            return await self._expire(
                checkpoint,
                f"decision budget exhausted: {', '.join(exhausted)}",
            )

        call_started = self._monotonic_clock()
        try:
            call = await self._proposal_producer.propose(context)
        except Exception as exc:
            return await self._fail(checkpoint, "Agent proposal", exc)
        if not isinstance(call, AgentCallResult):
            return await self._fail(
                checkpoint,
                "Agent proposal",
                TypeError("Agent must return AgentCallResult"),
            )
        observed_call_elapsed = self._monotonic_clock() - call_started
        usage = consume_agent_call(
            usage,
            elapsed_seconds=max(call.elapsed_seconds, observed_call_elapsed),
            cost_units=call.cost_units,
        )
        usage = usage.model_copy(
            update={
                "elapsed_seconds": max(
                    usage.elapsed_seconds,
                    self._monotonic_clock() - started,
                )
            }
        )
        proposal = call.proposal
        checkpoint = await self._advance(
            checkpoint,
            stage=DecisionCheckpointStage.PROPOSED,
            budget_usage=usage,
            proposal=proposal,
        )
        await self._trace(
            checkpoint,
            DecisionTraceKind.PROPOSED,
            payload={
                "proposal_type": proposal.proposal_type,
                "selected_action": proposal.selected_action,
                "revision": proposal.revision,
                "confidence": proposal.confidence,
                "agent_calls": usage.agent_calls,
                "consumed_cost_units": usage.consumed_cost_units,
            },
        )

        try:
            current_basis = await self._basis_provider.current_basis(request)
            validation_outcome = self._validator.validate(
                request,
                context,
                proposal,
                current_basis=current_basis,
                usage=usage,
            )
        except Exception as exc:
            return await self._fail(checkpoint, "Runtime validation", exc)
        validation = validation_outcome.validation
        if validation.status is DecisionValidationStatus.FAILED:
            status = (
                DecisionResultStatus.EXPIRED
                if any(
                    item.code
                    in {"request_expired", "stale_basis", "budget_exhausted"}
                    for item in validation.violations
                )
                else DecisionResultStatus.REJECTED
            )
            kind = (
                DecisionTraceKind.EXPIRED
                if status is DecisionResultStatus.EXPIRED
                else DecisionTraceKind.DENIED
            )
            await self._trace(
                checkpoint,
                DecisionTraceKind.VALIDATION_FAILED,
                validation_id=validation.validation_id,
                payload={
                    "violations": [item.code for item in validation.violations]
                },
            )
            return await self._complete(
                checkpoint,
                status=status,
                reason="; ".join(item.message for item in validation.violations),
                trace_kind=kind,
                validation=validation,
            )

        normalized = validation_outcome.normalized_effect
        if normalized is None:
            return await self._fail(
                checkpoint,
                "Runtime validation",
                DecisionInvariantError(
                    "passed validation did not produce a normalized effect"
                ),
            )
        validated = cast(
            ValidatedDecision[
                RequestPayloadT,
                ProposalPayloadT,
                EffectPayloadT,
            ],
            ValidatedDecision(
                request=request,
                proposal=proposal,
                validation=validation,
                normalized_effect=normalized,
            ),
        )
        checkpoint = await self._advance(
            checkpoint,
            stage=DecisionCheckpointStage.VALIDATED,
            validation=validation,
            validated_decision=validated,
        )
        await self._trace(
            checkpoint,
            DecisionTraceKind.VALIDATION_PASSED,
            validation_id=validation.validation_id,
            payload={"effect_fingerprint": normalized.effect_fingerprint},
        )
        await self._trace(
            checkpoint,
            DecisionTraceKind.GOVERNANCE_REQUESTED,
            validation_id=validation.validation_id,
            payload={
                "operation": normalized.operation,
                "target_type": normalized.target.target_type,
                "target_id": normalized.target.target_id,
                "risk": normalized.risk.value,
            },
        )
        try:
            resolution = await self._governance.evaluate(validated)
        except Exception as exc:
            return await self._fail(checkpoint, "Governance evaluation", exc)
        return await self._handle_governance(checkpoint, resolution)

    async def resume_review(
        self,
        request_id: UUID,
    ) -> DecisionCheckpoint[
        RequestPayloadT,
        ProposalPayloadT,
        EffectPayloadT,
    ]:
        checkpoint = await self._checkpoint_store.load(request_id)
        if checkpoint is None:
            raise DecisionResumeError("decision checkpoint does not exist")
        if checkpoint.stage is DecisionCheckpointStage.COMPLETED:
            return checkpoint
        if checkpoint.stage is not DecisionCheckpointStage.REVIEW_PENDING:
            raise DecisionResumeError("decision is not waiting for Human Review")
        validated = checkpoint.validated_decision
        receipt = checkpoint.governance_receipt
        if validated is None or receipt is None:
            raise DecisionInvariantError("review checkpoint is incomplete")
        try:
            stale_or_expired = await self._is_stale_or_expired(checkpoint)
        except Exception as exc:
            return await self._fail(checkpoint, "Runtime freshness check", exc)
        if stale_or_expired:
            return await self._expire(
                checkpoint,
                "decision became stale or expired while awaiting review",
            )
        try:
            resolution = await self._governance.resume_review(validated, receipt)
        except Exception as exc:
            return await self._fail(checkpoint, "Human Review finalization", exc)
        if (
            resolution.receipt.outcome
            is DecisionGovernanceOutcome.REVIEW_REQUIRED
        ):
            return checkpoint
        return await self._handle_governance(checkpoint, resolution)

    async def resume(
        self,
        request_id: UUID,
    ) -> DecisionCheckpoint[
        RequestPayloadT,
        ProposalPayloadT,
        EffectPayloadT,
    ]:
        """Resume without re-projecting context or invoking the producing Agent."""

        checkpoint = await self._checkpoint_store.load(request_id)
        if checkpoint is None:
            raise DecisionResumeError("decision checkpoint does not exist")
        if checkpoint.stage is DecisionCheckpointStage.COMPLETED:
            return checkpoint
        if checkpoint.stage is DecisionCheckpointStage.REVIEW_PENDING:
            return await self.resume_review(request_id)
        if checkpoint.stage in {
            DecisionCheckpointStage.REQUESTED,
            DecisionCheckpointStage.CONTEXT_PROJECTED,
            DecisionCheckpointStage.PROPOSED,
        }:
            return await self._expire(
                checkpoint,
                "decision interrupted before a validated effect; Agent is not recalled",
            )
        validated = checkpoint.validated_decision
        if validated is None:
            raise DecisionResumeError("decision checkpoint has no authorized effect")
        if checkpoint.stage is DecisionCheckpointStage.VALIDATED:
            try:
                resolution = await self._governance.evaluate(validated)
            except Exception as exc:
                return await self._fail(checkpoint, "Governance resume", exc)
            return await self._handle_governance(checkpoint, resolution)
        receipt = checkpoint.governance_receipt
        if receipt is None:
            raise DecisionResumeError("decision checkpoint has no Governance receipt")
        if checkpoint.stage is DecisionCheckpointStage.AUTHORIZED:
            approval = self._governance.restore_approval(validated, receipt)
            return await self._apply(checkpoint, approval)
        if checkpoint.stage not in {
            DecisionCheckpointStage.APPLYING,
            DecisionCheckpointStage.EFFECT_COMMITTED,
        }:
            raise DecisionResumeError(
                f"decision stage '{checkpoint.stage.value}' cannot safely resume"
            )
        approval = self._governance.restore_approval(validated, receipt)
        try:
            reconciliation = await self._applier.reconcile(validated, approval)
        except Exception as exc:
            return await self._fail(checkpoint, "Apply reconciliation", exc)
        if reconciliation.status is DecisionReconciliationStatus.COMMITTED:
            apply_receipt = reconciliation.apply_receipt
            if apply_receipt is None:
                raise DecisionInvariantError(
                    "committed reconciliation has no Apply receipt"
                )
            if checkpoint.stage is DecisionCheckpointStage.EFFECT_COMMITTED:
                persisted = checkpoint.commit_receipt
                if persisted is None or (
                    persisted.effect_fingerprint != apply_receipt.effect_fingerprint
                    or persisted.committed_state_fingerprint
                    != apply_receipt.committed_state_fingerprint
                    or persisted.result != apply_receipt.result
                ):
                    return await self._complete(
                        checkpoint,
                        status=DecisionResultStatus.FAILED,
                        reason=(
                            "Apply reconciliation failed closed: committed "
                            "read-back conflicts with persisted Commit receipt"
                        ),
                        trace_kind=DecisionTraceKind.FAILED,
                        reconciliation_status=DecisionReconciliationStatus.UNKNOWN,
                    )
                return await self._complete_applied(checkpoint, persisted)
            checkpoint = await self._mark_effect_committed(
                checkpoint, apply_receipt
            )
            return await self._complete_applied(checkpoint, apply_receipt)
        if checkpoint.stage is DecisionCheckpointStage.EFFECT_COMMITTED:
            return await self._complete(
                checkpoint,
                status=DecisionResultStatus.FAILED,
                reason=(
                    "Apply reconciliation failed closed: persisted Commit "
                    "receipt has no authoritative committed read-back"
                ),
                trace_kind=DecisionTraceKind.FAILED,
                reconciliation_status=DecisionReconciliationStatus.UNKNOWN,
            )
        if reconciliation.status is DecisionReconciliationStatus.EXPIRED:
            return await self._expire(checkpoint, reconciliation.reason)
        if reconciliation.status is DecisionReconciliationStatus.UNKNOWN:
            return await self._complete(
                checkpoint,
                status=DecisionResultStatus.FAILED,
                reason=("Apply reconciliation failed closed: " + reconciliation.reason),
                trace_kind=DecisionTraceKind.FAILED,
                reconciliation_status=DecisionReconciliationStatus.UNKNOWN,
            )
        try:
            stale_or_expired = await self._is_stale_or_expired(checkpoint)
        except Exception as exc:
            return await self._fail(checkpoint, "Runtime freshness reconciliation", exc)
        if stale_or_expired:
            return await self._expire(
                checkpoint,
                "uncommitted decision became stale or expired during interruption",
            )
        try:
            apply_receipt = await self._applier.resume_apply(validated, approval)
        except Exception as exc:
            return await self._fail(checkpoint, "Decision Apply resume", exc)
        checkpoint = await self._mark_effect_committed(checkpoint, apply_receipt)
        return await self._complete_applied(checkpoint, apply_receipt)

    async def _handle_governance(
        self,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        resolution: DecisionGovernanceResolution[ApprovalT],
    ) -> DecisionCheckpoint[
        RequestPayloadT,
        ProposalPayloadT,
        EffectPayloadT,
    ]:
        if not isinstance(resolution, DecisionGovernanceResolution):
            return await self._fail(
                checkpoint,
                "Governance evaluation",
                TypeError("Governance must return DecisionGovernanceResolution"),
            )
        receipt = resolution.receipt
        if receipt.outcome is DecisionGovernanceOutcome.DENY:
            return await self._complete(
                checkpoint,
                status=DecisionResultStatus.REJECTED,
                reason=receipt.reason,
                trace_kind=DecisionTraceKind.DENIED,
                governance_receipt=receipt,
            )
        if receipt.outcome is DecisionGovernanceOutcome.REVIEW_REQUIRED:
            checkpoint = await self._advance(
                checkpoint,
                stage=DecisionCheckpointStage.REVIEW_PENDING,
                governance_receipt=receipt,
            )
            await self._trace(
                checkpoint,
                DecisionTraceKind.REVIEW_REQUIRED,
                governance_decision_id=receipt.governance_decision_id,
                review_request_id=receipt.review_request_id,
                payload={"reason": receipt.reason},
            )
            self._inject_fault(DecisionFaultPoint.REVIEW_PENDING, checkpoint)
            return checkpoint
        if resolution.approval is None:
            return await self._fail(
                checkpoint,
                "Governance evaluation",
                DecisionInvariantError(
                    "allowed Governance decision has no approval binding"
                ),
            )
        checkpoint = await self._advance(
            checkpoint,
            stage=DecisionCheckpointStage.AUTHORIZED,
            governance_receipt=receipt,
        )
        await self._trace(
            checkpoint,
            DecisionTraceKind.AUTHORIZED,
            governance_decision_id=receipt.governance_decision_id,
            review_request_id=receipt.review_request_id,
            authorization_id=receipt.authorization_id,
            payload={"reason": receipt.reason},
        )
        self._inject_fault(DecisionFaultPoint.AUTHORIZED, checkpoint)
        return await self._apply(checkpoint, resolution.approval)

    async def _apply(
        self,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        approval: ApprovalT,
    ) -> DecisionCheckpoint[
        RequestPayloadT,
        ProposalPayloadT,
        EffectPayloadT,
    ]:
        validated = checkpoint.validated_decision
        if validated is None:
            raise DecisionInvariantError("authorized checkpoint has no validated effect")
        try:
            stale_or_expired = await self._is_stale_or_expired(checkpoint)
        except Exception as exc:
            return await self._fail(checkpoint, "Runtime freshness check", exc)
        if stale_or_expired:
            return await self._expire(
                checkpoint,
                "decision became stale or expired before apply",
            )
        checkpoint = await self._advance(
            checkpoint,
            stage=DecisionCheckpointStage.APPLYING,
        )
        receipt = checkpoint.governance_receipt
        await self._trace(
            checkpoint,
            DecisionTraceKind.APPLY_STARTED,
            governance_decision_id=(
                receipt.governance_decision_id if receipt is not None else None
            ),
            authorization_id=(receipt.authorization_id if receipt is not None else None),
            payload={
                "effect_fingerprint": (
                    validated.normalized_effect.effect_fingerprint
                )
            },
        )
        self._inject_fault(DecisionFaultPoint.APPLYING, checkpoint)
        try:
            apply_receipt = await self._applier.apply(validated, approval)
        except Exception as exc:
            return await self._fail(checkpoint, "Decision apply", exc)
        if (
            apply_receipt.effect_fingerprint
            != validated.normalized_effect.effect_fingerprint
        ):
            return await self._fail(
                checkpoint,
                "Decision apply",
                DecisionInvariantError("Apply receipt belongs to another effect"),
            )
        checkpoint = await self._mark_effect_committed(checkpoint, apply_receipt)
        self._inject_fault(DecisionFaultPoint.EFFECT_COMMITTED, checkpoint)
        return await self._complete_applied(checkpoint, apply_receipt)

    async def _mark_effect_committed(
        self,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        apply_receipt: DecisionApplyReceipt,
    ) -> DecisionCheckpoint[
        RequestPayloadT,
        ProposalPayloadT,
        EffectPayloadT,
    ]:
        validated = checkpoint.validated_decision
        if validated is None:
            raise DecisionInvariantError("Commit receipt has no validated effect")
        if apply_receipt.effect_fingerprint != validated.normalized_effect.effect_fingerprint:
            raise DecisionInvariantError("Commit receipt belongs to another effect")
        return await self._advance(
            checkpoint,
            stage=DecisionCheckpointStage.EFFECT_COMMITTED,
            commit_receipt=apply_receipt,
        )

    async def _complete_applied(
        self,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        apply_receipt: DecisionApplyReceipt,
    ) -> DecisionCheckpoint[
        RequestPayloadT,
        ProposalPayloadT,
        EffectPayloadT,
    ]:
        validated = checkpoint.validated_decision
        if validated is None:
            raise DecisionInvariantError("Apply completion has no validated effect")
        if apply_receipt.effect_fingerprint != validated.normalized_effect.effect_fingerprint:
            return await self._fail(
                checkpoint,
                "Decision apply",
                DecisionInvariantError("Apply receipt belongs to another effect"),
            )
        result = DecisionResult(
            request_id=checkpoint.request_id,
            proposal_id=validated.proposal.proposal_id,
            status=DecisionResultStatus.APPLIED,
            reason="validated and governed decision effect was applied",
            apply_receipt=apply_receipt,
            completed_at=self._clock(),
        )
        checkpoint = await self._advance(
            checkpoint,
            stage=DecisionCheckpointStage.COMPLETED,
            commit_receipt=apply_receipt,
            result=result,
        )
        await self._trace(
            checkpoint,
            DecisionTraceKind.APPLIED,
            payload={
                "result_id": str(result.result_id),
                "effect_fingerprint": apply_receipt.effect_fingerprint,
            },
        )
        return checkpoint

    async def _is_stale_or_expired(
        self,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
    ) -> bool:
        request = checkpoint.request
        if request.expires_at is not None and self._clock() >= request.expires_at:
            return True
        if budget_violations(request.budget, checkpoint.budget_usage):
            return True
        current = await self._basis_provider.current_basis(request)
        return current != request.basis

    async def _expire(
        self,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        reason: str,
    ) -> DecisionCheckpoint[
        RequestPayloadT,
        ProposalPayloadT,
        EffectPayloadT,
    ]:
        return await self._complete(
            checkpoint,
            status=DecisionResultStatus.EXPIRED,
            reason=reason,
            trace_kind=DecisionTraceKind.EXPIRED,
        )

    async def _fail(
        self,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        stage: str,
        exc: Exception,
    ) -> DecisionCheckpoint[
        RequestPayloadT,
        ProposalPayloadT,
        EffectPayloadT,
    ]:
        detail = str(exc) or exc.__class__.__name__
        return await self._complete(
            checkpoint,
            status=DecisionResultStatus.FAILED,
            reason=f"{stage} failed: {exc.__class__.__name__}: {detail}",
            trace_kind=DecisionTraceKind.FAILED,
        )

    async def _complete(
        self,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        *,
        status: DecisionResultStatus,
        reason: str,
        trace_kind: DecisionTraceKind,
        validation: DecisionValidation | None = None,
        governance_receipt: DecisionGovernanceReceipt | None = None,
        reconciliation_status: DecisionReconciliationStatus | None = None,
    ) -> DecisionCheckpoint[
        RequestPayloadT,
        ProposalPayloadT,
        EffectPayloadT,
    ]:
        proposal = checkpoint.proposal
        result = DecisionResult(
            request_id=checkpoint.request_id,
            proposal_id=proposal.proposal_id if proposal is not None else None,
            status=status,
            reason=reason,
            reconciliation_status=reconciliation_status,
            completed_at=self._clock(),
        )
        update: dict[str, object] = {"result": result}
        if validation is not None:
            update["validation"] = validation
        if governance_receipt is not None:
            update["governance_receipt"] = governance_receipt
        checkpoint = await self._advance(
            checkpoint,
            stage=DecisionCheckpointStage.COMPLETED,
            **update,
        )
        receipt = checkpoint.governance_receipt
        await self._trace(
            checkpoint,
            trace_kind,
            governance_decision_id=(
                receipt.governance_decision_id if receipt is not None else None
            ),
            review_request_id=(receipt.review_request_id if receipt is not None else None),
            authorization_id=(receipt.authorization_id if receipt is not None else None),
            payload={"status": status.value, "reason": reason},
        )
        return checkpoint

    def _inject_fault(
        self,
        point: DecisionFaultPoint,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
    ) -> None:
        if self._fault_injector is not None:
            self._fault_injector(point, checkpoint)

    async def _advance(
        self,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        *,
        stage: DecisionCheckpointStage,
        **updates: object,
    ) -> DecisionCheckpoint[
        RequestPayloadT,
        ProposalPayloadT,
        EffectPayloadT,
    ]:
        expected = checkpoint.revision
        values = {
            "revision": expected + 1,
            "stage": stage,
            "updated_at": self._clock(),
            **updates,
        }
        advanced = checkpoint.model_copy(update=values)
        await self._checkpoint_store.save(
            advanced,
            expected_revision=expected,
        )
        return advanced

    async def _trace(
        self,
        checkpoint: DecisionCheckpoint[
            RequestPayloadT,
            ProposalPayloadT,
            EffectPayloadT,
        ],
        kind: DecisionTraceKind,
        *,
        validation_id: UUID | None = None,
        governance_decision_id: UUID | None = None,
        review_request_id: UUID | None = None,
        authorization_id: UUID | None = None,
        payload: Mapping[str, JsonValue] | None = None,
    ) -> None:
        proposal = checkpoint.proposal
        correlation = checkpoint.request.correlation
        await self._trace_writer.record(
            DecisionTraceEvent(
                kind=kind,
                run_id=checkpoint.run_id,
                source=self.module_id,
                request_id=checkpoint.request_id,
                task_id=correlation.task_id,
                node_id=correlation.node_id,
                action_id=correlation.action_id,
                proposal_id=(proposal.proposal_id if proposal is not None else None),
                validation_id=validation_id,
                governance_decision_id=governance_decision_id,
                review_request_id=review_request_id,
                authorization_id=authorization_id,
                occurred_at=self._clock(),
                payload=payload or {},
            )
        )
