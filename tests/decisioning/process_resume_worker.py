from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from adaptive_agent_runtime.decisioning import (
    AgentCallResult,
    DecisionApplyReceipt,
    DecisionCheckpoint,
    DecisionFaultPoint,
    DecisionLifecycleCoordinator,
    DecisionProposal,
    DecisionReconciliation,
    DecisionReconciliationStatus,
    NormalizedDecisionEffect,
    DecisionRiskLevel,
    DecisionTraceEvent,
    PolicyAgentContextBuilder,
    RuntimeDecisionTraceWriter,
    RuntimeDecisionValidator,
    decision_fingerprint,
)
from adaptive_agent_runtime.governance import (
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernedDecisionApplier,
    HumanReviewDecision,
    ReviewOutcome,
    RuntimeDecisionGovernanceAdapter,
    RuntimeGovernanceEvaluator,
    default_governance_policy,
)
from adaptive_agent_runtime.persistence import SQLitePersistence
from tests.decisioning.fakes import (
    FakeAgent,
    FakeBasisProvider,
    FakeEffectPayload,
    FakeNormalizer,
    FakeProposalPayload,
    FakeRequestPayload,
    make_policy,
    make_request,
    make_sources,
)


CHECKPOINT_TYPE = DecisionCheckpoint[
    FakeRequestPayload,
    FakeProposalPayload,
    FakeEffectPayload,
]


class PersistentAgent(FakeAgent):
    def __init__(self, persistence: SQLitePersistence) -> None:
        super().__init__()
        self._persistence = persistence

    async def propose(self, context):  # type: ignore[no-untyped-def]
        with self._persistence.database.transaction() as cursor:
            cursor.execute(
                "INSERT INTO closeout_agent_invocations(request_id, calls) "
                "VALUES (?, 1) ON CONFLICT(request_id) DO UPDATE SET "
                "calls = calls + 1",
                (str(context.request_id),),
            )
        return await super().propose(context)


class RiskNormalizer(FakeNormalizer):
    def __init__(self, *, high_risk: bool) -> None:
        self._high_risk = high_risk

    def normalize(self, request, proposal):  # type: ignore[no-untyped-def]
        effect = super().normalize(request, proposal)
        if not self._high_risk:
            return effect
        return NormalizedDecisionEffect[FakeEffectPayload].create(
            payload=effect.payload,
            operation=effect.operation,
            target=effect.target,
            governance_scope=effect.governance_scope,
            risk=DecisionRiskLevel.HIGH,
            impact_score=effect.impact_score,
            reversible=effect.reversible,
            impact_description=effect.impact_description,
        )


def _initialize_ledger(persistence: SQLitePersistence, *, unknown: bool) -> None:
    with persistence.database.transaction() as cursor:
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS closeout_agent_invocations ("
            "request_id TEXT PRIMARY KEY, calls INTEGER NOT NULL)"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS closeout_effect_commits ("
            "effect_fingerprint TEXT PRIMARY KEY, payload_fingerprint TEXT NOT NULL, "
            "commit_count INTEGER NOT NULL, result_json TEXT NOT NULL, "
            "committed_at TEXT NOT NULL)"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS closeout_apply_attempts ("
            "effect_fingerprint TEXT PRIMARY KEY, attempts INTEGER NOT NULL)"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS closeout_worker_config ("
            "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
            "unknown_reconciliation INTEGER NOT NULL)"
        )
        cursor.execute(
            "INSERT INTO closeout_worker_config(singleton, unknown_reconciliation) "
            "VALUES (1, ?) ON CONFLICT(singleton) DO NOTHING",
            (1 if unknown else 0,),
        )


def _coordinator(
    persistence: SQLitePersistence,
    *,
    high_risk: bool,
    fault: DecisionFaultPoint | None,
):
    agent = PersistentAgent(persistence)
    evaluator = RuntimeGovernanceEvaluator(
        policy=default_governance_policy(),
        rule_evaluator=DeterministicRuleEvaluator(),
        confidence_evaluator=DeterministicConfidenceEvaluator(),
        review_service=persistence.human_review_service,
    )
    governance = RuntimeDecisionGovernanceAdapter[
        FakeRequestPayload,
        FakeProposalPayload,
        FakeEffectPayload,
    ](
        evaluator=evaluator,
        authorization_issuer=persistence.authorization_issuer,
        review_service=persistence.human_review_service,
    )

    async def apply_effect(normalized, permit):  # type: ignore[no-untyped-def]
        del permit
        effect_fingerprint = normalized.effect_fingerprint
        payload_fingerprint = decision_fingerprint(normalized.payload)
        with persistence.database.transaction() as cursor:
            cursor.execute(
                "INSERT INTO closeout_apply_attempts(effect_fingerprint, attempts) "
                "VALUES (?, 1) ON CONFLICT(effect_fingerprint) DO UPDATE SET "
                "attempts = attempts + 1",
                (effect_fingerprint,),
            )
            row = cursor.execute(
                "SELECT payload_fingerprint, result_json FROM closeout_effect_commits "
                "WHERE effect_fingerprint = ?",
                (effect_fingerprint,),
            ).fetchone()
            if row is not None:
                if row["payload_fingerprint"] != payload_fingerprint:
                    raise RuntimeError("effect fingerprint payload conflict")
                return json.loads(row["result_json"])
            result = {
                "effect_fingerprint": effect_fingerprint,
                "payload_fingerprint": payload_fingerprint,
                "value": normalized.payload.value,
            }
            cursor.execute(
                "INSERT INTO closeout_effect_commits "
                "(effect_fingerprint, payload_fingerprint, commit_count, result_json, committed_at) "
                "VALUES (?, ?, 1, ?, ?)",
                (
                    effect_fingerprint,
                    payload_fingerprint,
                    json.dumps(result, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return result

    async def reconcile(normalized):  # type: ignore[no-untyped-def]
        with persistence.database.reader() as cursor:
            row = cursor.execute(
                "SELECT result_json, committed_at FROM closeout_effect_commits "
                "WHERE effect_fingerprint = ?",
                (normalized.effect_fingerprint,),
            ).fetchone()
            config = cursor.execute(
                "SELECT unknown_reconciliation FROM closeout_worker_config WHERE singleton = 1"
            ).fetchone()
        if row is None:
            if config is not None and int(config["unknown_reconciliation"]) == 1:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason="test non-idempotent side effect cannot be classified",
                )
            return DecisionReconciliation(
                status=DecisionReconciliationStatus.NOT_COMMITTED,
                reason="no authoritative commit exists",
            )
        result = json.loads(row["result_json"])
        return DecisionReconciliation(
            status=DecisionReconciliationStatus.COMMITTED,
            reason="authoritative commit read back",
            apply_receipt=DecisionApplyReceipt(
                effect_fingerprint=normalized.effect_fingerprint,
                committed_state_fingerprint=decision_fingerprint(result),
                result=result,
                applied_at=datetime.fromisoformat(row["committed_at"]),
            ),
        )

    applier = GovernedDecisionApplier[
        FakeRequestPayload,
        FakeProposalPayload,
        FakeEffectPayload,
    ](
        executor=persistence.operation_executor,
        apply_effect=lambda payload: apply_effect(payload, None),
        apply_authorized_effect=apply_effect,
        reconcile_effect=reconcile,
    )

    def inject(point, checkpoint):  # type: ignore[no-untyped-def]
        del checkpoint
        if fault is point:
            os._exit(91)

    return DecisionLifecycleCoordinator[
        FakeRequestPayload,
        FakeProposalPayload,
        FakeEffectPayload,
        object,
    ](
        context_builder=PolicyAgentContextBuilder(),
        proposal_producer=agent,
        basis_provider=FakeBasisProvider(),
        validator=RuntimeDecisionValidator(
            normalizer=RiskNormalizer(high_risk=high_risk)
        ),
        governance=governance,
        applier=applier,
        checkpoint_store=persistence.create_decision_checkpoint_store(CHECKPOINT_TYPE),
        checkpoint_type=CHECKPOINT_TYPE,
        trace_writer=RuntimeDecisionTraceWriter(persistence.trace_sink),
        fault_injector=inject if fault is not None else None,
    )


async def _run(args: argparse.Namespace) -> None:
    persistence = SQLitePersistence(Path(args.database))
    try:
        _initialize_ledger(persistence, unknown=args.unknown)
        fault = DecisionFaultPoint(args.fault) if args.fault else None
        coordinator = _coordinator(
            persistence,
            high_risk=args.high_risk,
            fault=fault,
        )
        request = make_request()
        if args.action == "start":
            checkpoint = await coordinator.run(
                request,
                sources=make_sources(),
                policy=make_policy(),
            )
        else:
            checkpoint = await persistence.create_decision_checkpoint_store(
                CHECKPOINT_TYPE
            ).load(request.request_id)
            if checkpoint is None:
                raise RuntimeError("checkpoint does not exist")
            if args.action == "approve":
                receipt = checkpoint.governance_receipt
                if receipt is None or receipt.review_request_id is None:
                    raise RuntimeError("pending review is missing")
                persistence.human_review_service.resolve(
                    receipt.review_request_id,
                    HumanReviewDecision(
                        outcome=ReviewOutcome.APPROVE,
                        reviewer_id="closeout-test",
                        rationale="approved for process resume test",
                    ),
                )
                checkpoint = await coordinator.resume_review(request.request_id)
            else:
                checkpoint = await coordinator.resume(request.request_id)
        print(checkpoint.model_dump_json())
    finally:
        persistence.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--action", choices=("start", "resume", "approve"), required=True)
    parser.add_argument("--fault", choices=tuple(item.value for item in DecisionFaultPoint))
    parser.add_argument("--high-risk", action="store_true")
    parser.add_argument("--unknown", action="store_true")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
