"""Deterministic context-pressure and governed-memory scenario executors."""

from __future__ import annotations

from collections.abc import Mapping

from adaptive_agent_runtime.decisioning import decision_fingerprint
from applications.ecommerce_support.policies import DomainPolicyError
from applications.governance_scenario_suite.contracts import (
    AuditEventType,
    MemoryFixture,
    ReasonCode,
    ScenarioProfile,
    ScenarioVerdict,
)
from applications.governance_scenario_suite.evidence import (
    EvidenceSource,
    MemoryEvidenceView,
    ScenarioExecution,
)
from applications.governance_scenario_suite.runner import ScenarioRuntimeContext


class FullAARContextMemoryExecutor:
    profile = ScenarioProfile.FULL_AAR

    def execute(self, context: object) -> ScenarioExecution:
        if not isinstance(context, ScenarioRuntimeContext):
            raise TypeError("context-memory executor requires ScenarioRuntimeContext")
        if context.scenario.family_id == "C1_AUTHORITATIVE_CONSTRAINT":
            return _authoritative_constraint(context, enforce=True)
        if context.scenario.family_id in {"M1_MEMORY_SCOPE", "M2_CONDITIONAL_MEMORY"}:
            return _memory_recall(context, filter_scope=True)
        raise ValueError("unsupported context-memory family")


class NoAuthoritativeConstraintExecutor:
    profile = ScenarioProfile.NO_AUTHORITATIVE_CONSTRAINT_CHECK

    def execute(self, context: object) -> ScenarioExecution:
        if not isinstance(context, ScenarioRuntimeContext):
            raise TypeError("constraint ablation requires ScenarioRuntimeContext")
        if context.scenario.family_id != "C1_AUTHORITATIVE_CONSTRAINT":
            safe = FullAARContextMemoryExecutor()
            return safe.execute(context)
        return _authoritative_constraint(context, enforce=False)


class NoMemoryScopeFilterExecutor:
    profile = ScenarioProfile.NO_MEMORY_SCOPE_FILTER

    def execute(self, context: object) -> ScenarioExecution:
        if not isinstance(context, ScenarioRuntimeContext):
            raise TypeError("memory ablation requires ScenarioRuntimeContext")
        if context.scenario.family_id not in {"M1_MEMORY_SCOPE", "M2_CONDITIONAL_MEMORY"}:
            return FullAARContextMemoryExecutor().execute(context)
        return _memory_recall(context, filter_scope=False)


def _authoritative_constraint(
    context: ScenarioRuntimeContext,
    *,
    enforce: bool,
) -> ScenarioExecution:
    projected = _bounded_context(context.scenario.conversation, budget=300)
    proposal = context.model.propose(projected)
    context.composition.store.append_audit(
        occurred_at=context.scenario.clock,
        event_type=AuditEventType.PROPOSAL_RECORDED,
        effect_fingerprint=decision_fingerprint(proposal),
        idempotency_key=proposal.idempotency_key,
    )
    if not enforce:
        assert proposal.order_id is not None
        assert proposal.amount_cents is not None
        assert proposal.idempotency_key is not None
        order = context.composition.store.load_order_authoritative(proposal.order_id)
        if order is None:
            raise ValueError("constraint scenario order is missing")
        if proposal.amount_cents <= (
            order.paid_amount_cents - order.refunded_amount_cents
        ):
            context.composition.tools.request_refund(context.principal, proposal)
            return _result(context, ScenarioVerdict.ALLOW)
        context.composition.gateway.refund(
            order_id=proposal.order_id,
            amount_cents=proposal.amount_cents,
            idempotency_key=proposal.idempotency_key,
            now=context.scenario.clock,
        )
        return _result(context, ScenarioVerdict.ALLOW)
    try:
        context.composition.tools.request_refund(context.principal, proposal)
    except DomainPolicyError as exc:
        context.composition.store.append_audit(
            occurred_at=context.scenario.clock,
            event_type=AuditEventType.AUTHORITATIVE_CONSTRAINT_REJECTED,
            reason_code=exc.reason_code,
            effect_fingerprint=decision_fingerprint(proposal),
            idempotency_key=proposal.idempotency_key,
        )
        return _result(context, ScenarioVerdict.REJECT, exc.reason_code)
    return _result(context, ScenarioVerdict.ALLOW)


def _memory_recall(
    context: ScenarioRuntimeContext,
    *,
    filter_scope: bool,
) -> ScenarioExecution:
    stored = context.scenario.initial_authoritative_state.memories
    candidates = tuple(
        item
        for item in stored
        if not filter_scope or _in_scope(context, item)
    )
    bundle = tuple(
        item
        for item in candidates
        if _conditions_match(item, context.scenario.context_facts)
    )
    projected = "\n".join(str(item.content) for item in bundle)
    context.model.propose(projected or "No applicable long-term preferences.")
    memory = MemoryEvidenceView(
        candidate_canaries=tuple(item.canary for item in candidates if item.canary),
        bundle_canaries=tuple(item.canary for item in bundle if item.canary),
        memory_keys=tuple(item.memory_key for item in bundle),
        revision_count=len(stored),
    )
    return _result(context, ScenarioVerdict.ALLOW, memory=memory)


def _in_scope(context: ScenarioRuntimeContext, item: MemoryFixture) -> bool:
    return (
        item.tenant_id == context.principal.tenant_id
        and (
            item.subject_user_id is None
            or item.subject_user_id == context.principal.user_id
        )
    )


def _conditions_match(item: MemoryFixture, facts: Mapping[str, object]) -> bool:
    return all(facts.get(key) == value for key, value in item.condition_facts.items())


def _bounded_context(turns: tuple[str, ...], *, budget: int) -> str:
    joined = "\n".join(turns)
    return joined[-budget:]


def _result(
    context: ScenarioRuntimeContext,
    decision: ScenarioVerdict,
    reason: ReasonCode | None = None,
    *,
    memory: MemoryEvidenceView | None = None,
) -> ScenarioExecution:
    sources = {EvidenceSource.AUDIT, EvidenceSource.MODEL_CONTEXT}
    if memory is not None:
        sources.add(EvidenceSource.MEMORY)
    return ScenarioExecution(
        decision=decision,
        reason_code=reason,
        model_contexts=context.model.contexts,
        memory=memory or MemoryEvidenceView(),
        available_sources=frozenset(sources),
        model_call_count=context.model.calls,
    )
