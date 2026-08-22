"""Plain Agent baseline and full non-recovery executor routing."""

from __future__ import annotations

from applications.governance_scenario_suite.context_memory_executors import (
    FullAARContextMemoryExecutor,
    NoMemoryScopeFilterExecutor,
)
from applications.governance_scenario_suite.contracts import (
    ScenarioProfile,
    ScenarioVerdict,
)
from applications.governance_scenario_suite.evidence import (
    EvidenceSource,
    ScenarioExecution,
)
from applications.governance_scenario_suite.executors import (
    FullAARPermissionExecutor,
    NoExactEffectBindingExecutor,
)
from applications.governance_scenario_suite.runner import ScenarioRuntimeContext


class FullAARScenarioExecutor:
    profile = ScenarioProfile.FULL_AAR

    def __init__(self) -> None:
        self._permission = FullAARPermissionExecutor()
        self._context_memory = FullAARContextMemoryExecutor()

    def execute(self, context: object) -> ScenarioExecution:
        if not isinstance(context, ScenarioRuntimeContext):
            raise TypeError("full executor requires ScenarioRuntimeContext")
        if context.scenario.family_id.startswith("P"):
            return self._permission.execute(context)
        return self._context_memory.execute(context)


class PlainAgentExecutor:
    """Deliberately under-governed baseline; never used by production composition."""

    profile = ScenarioProfile.PLAIN_AGENT

    def __init__(self) -> None:
        self._safe = FullAARScenarioExecutor()
        self._no_binding = NoExactEffectBindingExecutor()
        self._no_memory_scope = NoMemoryScopeFilterExecutor()

    def execute(self, context: object) -> ScenarioExecution:
        if not isinstance(context, ScenarioRuntimeContext):
            raise TypeError("plain baseline requires ScenarioRuntimeContext")
        family = context.scenario.family_id
        if family == "P2_EFFECT_BINDING":
            return self._no_binding.execute(context)
        if family in {"M1_MEMORY_SCOPE", "M2_CONDITIONAL_MEMORY"}:
            return self._no_memory_scope.execute(context)
        if family in {"P1_RESOURCE_SCOPE", "P5_PROMPT_INJECTION"}:
            return self._unscoped_read(context)
        return self._safe.execute(context)

    @staticmethod
    def _unscoped_read(context: ScenarioRuntimeContext) -> ScenarioExecution:
        proposal = context.model.propose(
            f"User goal: {context.scenario.user_goal}\n" + "\n".join(context.scenario.conversation)
        )
        assert proposal.order_id is not None
        order = context.composition.store.load_order_authoritative(proposal.order_id)
        projected: list[str] = list(context.model.contexts)
        if order is not None:
            projected.append(
                f"order_id={order.order_id}; address_ref={order.address_ref}; description={order.description}"
            )
            for payment in context.composition.store.snapshot().payments:
                if payment.order_id == order.order_id:
                    projected.append(
                        f"payment={payment.display_reference}; token={payment.sensitive_canary}"
                    )
        return ScenarioExecution(
            decision=ScenarioVerdict.ALLOW,
            model_contexts=tuple(projected),
            available_sources=frozenset(
                {EvidenceSource.AUDIT, EvidenceSource.MODEL_CONTEXT}
            ),
            model_call_count=context.model.calls,
        )
