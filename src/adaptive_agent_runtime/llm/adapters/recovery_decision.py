"""Recovery Agent adapter into the Runtime-owned Decision Lifecycle."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from time import monotonic

from adaptive_agent_runtime.decisioning import (
    AgentCallResult,
    AgentContext,
    DecisionEvidenceReference,
    DecisionGovernanceScope,
    DecisionProducer,
    DecisionProposal,
    DecisionRequest,
    DecisionRiskLevel,
    NormalizedDecisionEffect,
    decision_fingerprint,
)
from adaptive_agent_runtime.llm.capabilities.contracts import (
    RecoveryProposalCapability,
)
from adaptive_agent_runtime.llm.capabilities.models import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    EvidenceReference,
    RecoveryActionDraftKind,
    RecoveryDraft,
    RecoveryFailureKindDraft,
    RecoveryNodeCandidate,
    RecoveryProposalRequest,
)
from adaptive_agent_runtime.llm.capabilities.validation import (
    CapabilityDraftValidator,
)
from adaptive_agent_runtime.llm.models import InferenceCorrelation
from adaptive_agent_runtime.orchestration.models import TaskNode
from adaptive_agent_runtime.orchestration.recovery import (
    FailureAnalysis,
    FailureKind,
    RecoveryAction,
    RecoveryActionType,
    RecoveryPlan,
    apply_recovery_plan,
    stable_recovery_id,
)
from adaptive_agent_runtime.orchestration.recovery_decision import (
    RECOVERY_APPLY_OPERATION,
    RECOVERY_DECISION_TYPE,
    AbortRecoveryEffect,
    AddRecoveryNodeEffect,
    RecoveryDecisionEffect,
    RecoveryDecisionPayload,
    RecoveryExecutionPolicy,
    ReplaceStrategyRecoveryEffect,
    RetryNodeRecoveryEffect,
    RewireDependencyRecoveryEffect,
)


RECOVERY_INPUT_SOURCE_TYPE = "recovery_input"
_CONTROL_DIRECTIVE = re.compile(
    r"(?i)(?:^|\s)(?:execute_tool|runtime_state_change|state_patch|"
    r"runtime\.apply|tool\.call)\s*\("
)


def build_recovery_proposal_request(
    payload: RecoveryDecisionPayload,
    evidence: tuple[DecisionEvidenceReference, ...],
) -> RecoveryProposalRequest:
    """Project a finite Graph summary without exposing live Runtime objects."""

    failed = payload.graph.get_node(payload.failed_node_id)
    observation = failed.observation
    if observation is None or observation.succeeded or observation.error is None:
        raise ValueError("Recovery projection requires a failed Observation")
    candidates = tuple(
        RecoveryNodeCandidate(
            node_ref=payload.node_ref_for(node.node_id),
            goal=node.goal,
            status=node.status.value,
            dependency_refs=tuple(
                payload.node_ref_for(item) for item in node.dependencies
            ),
            strategy_id=node.strategy_id,
        )
        for node in sorted(payload.graph.nodes, key=lambda item: str(item.node_id))
    )
    return RecoveryProposalRequest(
        task=payload.task_description,
        failed_node_ref=payload.node_ref_for(payload.failed_node_id),
        failed_goal=failed.goal,
        error=observation.error,
        graph_nodes=candidates,
        available_strategies=payload.available_strategy_ids,
        allowed_recovery_actions=tuple(
            RecoveryActionDraftKind(item.value)
            for item in payload.allowed_recovery_actions
        ),
        prior_attempts=payload.prior_attempts,
        max_attempts=payload.execution_policy.max_recovery_attempts,
        evidence=tuple(
            EvidenceReference(
                reference_id=item.evidence_id,
                kind=item.kind,
                summary=item.summary,
                reliability=item.reliability,
            )
            for item in evidence
        ),
    )


class RecoveryRequestAdapter:
    """Read only the isolated recovery block supplied by Context Projection."""

    module_id = "llm.adapter.recovery.request"

    def to_request(self, context: AgentContext) -> RecoveryProposalRequest:
        blocks = tuple(
            item
            for item in context.blocks
            if item.source_type == RECOVERY_INPUT_SOURCE_TYPE
        )
        if len(blocks) != 1:
            raise ValueError("Recovery Agent context requires one recovery_input block")
        content = blocks[0].content
        if not isinstance(content, Mapping):
            raise ValueError("recovery_input content must be an object")
        return RecoveryProposalRequest.model_validate(dict(content))


class RecoveryDecisionProposalProducer:
    """Make one bounded Agent call; its Draft cannot execute recovery."""

    module_id = "llm.adapter.recovery.proposal_producer"

    def __init__(
        self,
        *,
        capability: RecoveryProposalCapability,
        execution_policy: RecoveryExecutionPolicy,
        correlation: InferenceCorrelation,
        request_adapter: RecoveryRequestAdapter | None = None,
    ) -> None:
        self._capability = capability
        self._execution_policy = execution_policy
        self._correlation = correlation
        self._request_adapter = request_adapter or RecoveryRequestAdapter()
        self._called_request_ids: set[object] = set()

    async def propose(self, context: AgentContext) -> AgentCallResult[RecoveryDraft]:
        if context.request_id in self._called_request_ids:
            raise RuntimeError("RecoveryExecutionPolicy permits one Agent call")
        self._called_request_ids.add(context.request_id)
        request = self._request_adapter.to_request(context)
        started = monotonic()
        turn = await asyncio.wait_for(
            self._capability.propose_recovery(
                request,
                invocation=CapabilityInvocationMetadata(
                    correlation=self._correlation,
                    trace_attributes={
                        "operation": "recovery.propose",
                        "recovery_timeout_seconds": (
                            self._execution_policy.timeout_seconds
                        ),
                        "recovery_max_agent_calls": (
                            self._execution_policy.max_agent_calls
                        ),
                    },
                ),
            ),
            timeout=self._execution_policy.timeout_seconds,
        )
        elapsed = monotonic() - started
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Recovery Agent cannot execute ToolIntents")
        if turn.result is None or not isinstance(turn.result, RecoveryDraft):
            raise RuntimeError("Recovery Agent must return only RecoveryDraft")
        draft = turn.result
        return AgentCallResult(
            proposal=DecisionProposal[RecoveryDraft](
                request_id=context.request_id,
                proposal_type=context.decision_type,
                producer=DecisionProducer(
                    producer_id=self._capability.capability_id,
                    capability="recovery_proposal",
                    implementation_version="phase-1-recovery",
                ),
                input_snapshot_fingerprint=context.basis_fingerprint,
                context_fingerprint=context.context_fingerprint,
                revision=0,
                selected_action=RECOVERY_APPLY_OPERATION,
                payload=draft,
                rationale=draft.rationale,
                evidence_refs=draft.evidence_reference_ids,
                confidence=draft.confidence,
            ),
            elapsed_seconds=elapsed,
            cost_units=0.0,
        )


class RecoveryEffectNormalizer:
    """Validate a Draft and project one Runtime-owned, typed Recovery Effect."""

    module_id = "llm.adapter.recovery.effect_normalizer"

    def __init__(
        self,
        *,
        draft_validator: CapabilityDraftValidator | None = None,
    ) -> None:
        self._draft_validator = draft_validator or CapabilityDraftValidator()

    def normalize(
        self,
        request: DecisionRequest[RecoveryDecisionPayload],
        proposal: DecisionProposal[RecoveryDraft],
    ) -> NormalizedDecisionEffect[RecoveryDecisionEffect]:
        if request.decision_type != RECOVERY_DECISION_TYPE:
            raise ValueError("Recovery normalizer received another decision type")
        draft = proposal.payload
        projected_request = build_recovery_proposal_request(
            request.payload,
            request.evidence,
        )
        self._draft_validator.validate_recovery_proposal(
            projected_request,
            draft,
        )
        self._reject_control_directives(draft)
        plan = self._to_plan(request, draft)
        preview = apply_recovery_plan(request.payload.graph, plan)
        effect = RecoveryDecisionEffect(
            graph_id=request.payload.graph.graph_id,
            graph_version_before=request.payload.graph.version,
            graph_version_after=preview.version,
            source_draft_fingerprint=decision_fingerprint(draft),
            basis_fingerprint=request.basis.snapshot_fingerprint,
            hypothesis=draft.hypothesis,
            alternatives=draft.alternatives,
            agent_confidence=draft.confidence,
            plan=plan,
            effect=self._typed_effect(plan),
        )
        risk, impact, reversible, description, operation = self._runtime_risk(
            plan.actions[0].action_type
        )
        return NormalizedDecisionEffect[RecoveryDecisionEffect].create(
            payload=effect,
            operation=operation,
            target=request.target,
            governance_scope=DecisionGovernanceScope.STATE,
            risk=risk,
            impact_score=impact,
            reversible=reversible,
            impact_description=description,
        )

    @staticmethod
    def _to_plan(
        request: DecisionRequest[RecoveryDecisionPayload],
        draft: RecoveryDraft,
    ) -> RecoveryPlan:
        payload = request.payload
        selected = draft.selected_action
        action_type = RecoveryActionType(selected.kind.value)
        failed_node_id = payload.node_id_for(selected.target_node_ref)
        action_values: dict[str, object] = {
            "action_type": action_type,
            "node_id": failed_node_id,
            "reason": selected.reason,
        }
        if action_type is RecoveryActionType.REPLACE_STRATEGY:
            action_values["replacement_strategy_id"] = (
                selected.replacement_strategy_id
            )
        elif action_type is RecoveryActionType.ADD_RECOVERY_NODE:
            assert selected.recovery_node is not None
            node = selected.recovery_node
            action_values["recovery_node"] = TaskNode(
                node_id=stable_recovery_id(
                    "agent-recovery-node",
                    request.request_id,
                    payload.graph.graph_id,
                    node.node_key,
                ),
                goal=node.goal,
                dependencies=tuple(
                    payload.node_id_for(item) for item in node.dependency_keys
                ),
                expected_output=node.expected_output,
                strategy_id=node.requested_strategy_id,
            )
        elif action_type is RecoveryActionType.REWIRE_DEPENDENCY:
            assert selected.dependent_node_ref is not None
            assert selected.old_dependency_ref is not None
            assert selected.new_dependency_ref is not None
            action_values.update(
                {
                    "dependent_node_id": payload.node_id_for(
                        selected.dependent_node_ref
                    ),
                    "old_dependency_id": payload.node_id_for(
                        selected.old_dependency_ref
                    ),
                    "new_dependency_id": payload.node_id_for(
                        selected.new_dependency_ref
                    ),
                }
            )
        evidence_by_id = {item.evidence_id: item for item in request.evidence}
        evidence = tuple(
            evidence_by_id[item].summary
            for item in draft.evidence_reference_ids
        )
        kind = FailureKind(RecoveryFailureKindDraft(draft.failure_kind).value)
        analysis = FailureAnalysis(
            analysis_id=stable_recovery_id(
                "agent-failure-analysis",
                request.request_id,
                payload.failed_action_id,
            ),
            node_id=payload.failed_node_id,
            action_id=payload.failed_action_id,
            kind=kind,
            retryable=(
                action_type is not RecoveryActionType.ABORT
                and kind is not FailureKind.NON_RECOVERABLE
            ),
            confidence=draft.confidence,
            evidence=evidence,
        )
        return RecoveryPlan(
            plan_id=stable_recovery_id(
                "agent-recovery-plan",
                request.request_id,
                decision_fingerprint(draft),
            ),
            analysis=analysis,
            actions=(RecoveryAction.model_validate(action_values),),
            attempt_number=(
                min(
                    payload.prior_attempts + 1,
                    payload.execution_policy.max_recovery_attempts,
                )
                if action_type is RecoveryActionType.ABORT
                else payload.prior_attempts + 1
            ),
            max_attempts=payload.execution_policy.max_recovery_attempts,
        )

    @staticmethod
    def _typed_effect(
        plan: RecoveryPlan,
    ) -> (
        RetryNodeRecoveryEffect
        | ReplaceStrategyRecoveryEffect
        | AddRecoveryNodeEffect
        | RewireDependencyRecoveryEffect
        | AbortRecoveryEffect
    ):
        action = plan.actions[0]
        if action.action_type is RecoveryActionType.RETRY_NODE:
            return RetryNodeRecoveryEffect(node_id=action.node_id)
        if action.action_type is RecoveryActionType.REPLACE_STRATEGY:
            assert action.replacement_strategy_id is not None
            return ReplaceStrategyRecoveryEffect(
                node_id=action.node_id,
                strategy_id=action.replacement_strategy_id,
            )
        if action.action_type is RecoveryActionType.ADD_RECOVERY_NODE:
            assert action.recovery_node is not None
            return AddRecoveryNodeEffect(
                failed_node_id=action.node_id,
                recovery_node=action.recovery_node,
            )
        if action.action_type is RecoveryActionType.REWIRE_DEPENDENCY:
            assert action.dependent_node_id is not None
            assert action.old_dependency_id is not None
            assert action.new_dependency_id is not None
            return RewireDependencyRecoveryEffect(
                dependent_node_id=action.dependent_node_id,
                old_dependency_id=action.old_dependency_id,
                new_dependency_id=action.new_dependency_id,
            )
        return AbortRecoveryEffect(
            failed_node_id=action.node_id,
            reason=action.reason,
        )

    @staticmethod
    def _runtime_risk(
        action_type: RecoveryActionType,
    ) -> tuple[DecisionRiskLevel, float, bool, str, str]:
        """Derive risk only from normalized impact, never Agent confidence."""

        if action_type is RecoveryActionType.RETRY_NODE:
            return (
                DecisionRiskLevel.MEDIUM,
                0.2,
                True,
                "Reopen the failed node without changing its execution strategy.",
                "recovery.retry",
            )
        if action_type is RecoveryActionType.REPLACE_STRATEGY:
            return (
                DecisionRiskLevel.MEDIUM,
                0.45,
                True,
                "Reopen the failed node with a Runtime-available strategy.",
                "recovery.change_strategy",
            )
        if action_type is RecoveryActionType.ADD_RECOVERY_NODE:
            return (
                DecisionRiskLevel.HIGH,
                0.75,
                True,
                "Add a bounded recovery node and reroute unresolved dependents.",
                "graph.mutate.recovery_node",
            )
        if action_type is RecoveryActionType.REWIRE_DEPENDENCY:
            return (
                DecisionRiskLevel.HIGH,
                0.7,
                True,
                "Rewire an unresolved dependency in the active task graph.",
                "graph.mutate.recovery_dependency",
            )
        return (
            DecisionRiskLevel.HIGH,
            0.8,
            False,
            "Terminate recovery and preserve the failed execution outcome.",
            "recovery.abort",
        )

    @staticmethod
    def _reject_control_directives(draft: RecoveryDraft) -> None:
        values = [draft.hypothesis, draft.rationale, *draft.alternatives]
        selected = draft.selected_action
        values.append(selected.reason)
        if selected.recovery_node is not None:
            values.extend(
                [
                    selected.recovery_node.goal,
                    selected.recovery_node.expected_output,
                ]
            )
        if any(_CONTROL_DIRECTIVE.search(value) for value in values):
            raise ValueError("Recovery Draft contains a reserved Runtime directive")
