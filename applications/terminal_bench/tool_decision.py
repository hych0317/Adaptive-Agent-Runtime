"""Bind terminal drafts to the existing Tool Invocation Decision Lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID, uuid5, NAMESPACE_URL

from pydantic import JsonValue

from adaptive_agent_runtime.core import ActionRequest, AgentState, TraceSink
from adaptive_agent_runtime.decisioning import (
    ContextProjectionPolicy,
    ContextSensitivity,
    DecisionApplyReceipt,
    DecisionBasis,
    DecisionBudget,
    DecisionCheckpoint,
    DecisionCheckpointStage,
    DecisionCheckpointStore,
    DecisionConstraint,
    DecisionCorrelation,
    DecisionLifecycleCoordinator,
    DecisionReconciliation,
    DecisionReconciliationStatus,
    DecisionRequest,
    DecisionResultStatus,
    DecisionTarget,
    InMemoryDecisionCheckpointStore,
    NormalizedDecisionEffect,
    PolicyAgentContextBuilder,
    ProjectionSource,
    ProjectionSources,
    RuntimeDecisionTraceWriter,
    RuntimeDecisionValidator,
    decision_fingerprint,
)
from adaptive_agent_runtime.governance import (
    DecisionGovernanceBinding,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernanceEvaluator,
    GovernanceTarget,
    GovernedDecisionApplier,
    GovernedOperationExecutor,
    HumanReviewService,
    InMemoryHumanReviewService,
    RuntimeCommitPermit,
    RuntimeDecisionGovernanceAdapter,
    RuntimeGovernanceEvaluator,
    default_governance_policy,
    governance_fingerprint,
)
from adaptive_agent_runtime.llm import (
    BoundToolInvocationProposalProducer,
    ToolInvocationEffectNormalizer,
    build_tool_invocation_agent_input,
)
from adaptive_agent_runtime.tool_ecosystem import (
    TOOL_INVOCATION_DECISION_TYPE,
    TOOL_INVOCATION_INPUT_SOURCE_TYPE,
    TOOL_INVOCATION_OPERATION,
    CapabilityRequirement,
    PermitBoundToolExecutor,
    RetryPolicy,
    ToolExecutionPolicy,
    ToolInvocation,
    ToolInvocationDecisionOutcome,
    ToolInvocationDecisionPayload,
    ToolInvocationEffect,
    ToolInvocationProposalDraft,
    ToolCorrelation,
    ToolObservation,
    ToolProviderMetadata,
    stable_tool_invocation_decision_id,
    tool_invocation_candidate_set_fingerprint,
    tool_invocation_fingerprint,
)

from applications.terminal_bench.contracts import TerminalTrialJournal
from applications.terminal_bench.models import (
    TERMINAL_COMMAND_CAPABILITY,
    TerminalCommandIntent,
    TerminalExecutionPolicy,
)
from applications.terminal_bench.tools import terminal_tool_observation_from_result


_TERMINAL_REQUIREMENT_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "adaptive-agent-runtime/terminal-sequential/requirement",
)


@dataclass(frozen=True)
class TerminalToolDecisionResult:
    request_id: UUID
    invocation: ToolInvocation
    status: str
    effect_fingerprint: str | None = None
    execution_in_doubt: bool = False
    outcome: ToolInvocationDecisionOutcome | None = None
    reason: str | None = None


def _invocation_basis(payload: ToolInvocationDecisionPayload) -> DecisionBasis:
    return DecisionBasis(
        snapshot_fingerprint=decision_fingerprint(
            {
                "invocation": payload.invocation,
                "requirement": payload.requirement,
                "candidate_set_fingerprint": payload.candidate_set_fingerprint,
                "provider_metadata_fingerprint": payload.provider_metadata_fingerprint,
                "selection_fingerprint": payload.selection_fingerprint,
                "proposal_fingerprint": payload.proposal_fingerprint,
                "exact_argument_constraints": payload.exact_argument_constraints,
                "state_revision": payload.state_revision,
                "privileged": payload.privileged,
            }
        ),
        state_revision=payload.state_revision,
    )


class _TerminalInvocationBasisProvider:
    module_id = "terminal_bench.tool_decision.basis"

    def __init__(self, basis: DecisionBasis) -> None:
        self._basis = basis

    async def current_basis(self, request: DecisionRequest[Any]) -> DecisionBasis:
        del request
        return self._basis


class TerminalToolInvocationDecisionHandler:
    """One generic Tool Invocation decision; this is not a second lifecycle."""

    module_id = "terminal_bench.tool_invocation_decision_handler"

    def __init__(
        self,
        *,
        provider_metadata: ToolProviderMetadata,
        executor: PermitBoundToolExecutor,
        operation_executor: GovernedOperationExecutor,
        authorization_issuer: GovernanceAuthorizationIssuer,
        trace_sink: TraceSink,
        journal: TerminalTrialJournal,
        policy: TerminalExecutionPolicy,
        governance: GovernanceEvaluator | None = None,
        reviews: HumanReviewService | None = None,
        checkpoint_store: DecisionCheckpointStore[
            ToolInvocationDecisionPayload,
            ToolInvocationProposalDraft,
            ToolInvocationEffect,
        ]
        | None = None,
    ) -> None:
        if provider_metadata.capability_id != TERMINAL_COMMAND_CAPABILITY:
            raise ValueError("terminal Provider implements the wrong capability")
        self._metadata = provider_metadata
        self._executor = executor
        self._operation_executor = operation_executor
        self._issuer = authorization_issuer
        self._trace_sink = trace_sink
        self._journal = journal
        self._policy = policy
        self._reviews = reviews or InMemoryHumanReviewService()
        self._governance = governance or RuntimeGovernanceEvaluator(
            policy=default_governance_policy(),
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=self._reviews,
        )
        self._checkpoint_store = (
            checkpoint_store
            or InMemoryDecisionCheckpointStore[
                ToolInvocationDecisionPayload,
                ToolInvocationProposalDraft,
                ToolInvocationEffect,
            ]()
        )

    async def handle(
        self,
        *,
        proposal: ToolInvocationProposalDraft,
        action: ActionRequest,
        state: AgentState,
        producer_id: str,
    ) -> TerminalToolDecisionResult:
        if proposal.capability_id != TERMINAL_COMMAND_CAPABILITY:
            raise ValueError("terminal action requested another capability")
        intent = TerminalCommandIntent.model_validate(
            {**dict(proposal.arguments), "call_key": proposal.call_key}
        )
        if intent.trial_id != self._journal.trial_id:
            raise ValueError("terminal action changed trial scope")
        requirement = CapabilityRequirement(
            requirement_id=uuid5(
                _TERMINAL_REQUIREMENT_NAMESPACE,
                f"{intent.trial_id}|{TERMINAL_COMMAND_CAPABILITY}",
            ),
            capability_id=TERMINAL_COMMAND_CAPABILITY,
            required_provider_tags=("trial_scoped",),
        )
        invocation_id = stable_tool_invocation_decision_id(
            "terminal-invocation",
            state.run_id,
            action.action_id,
            proposal.call_key,
        )
        invocation = ToolInvocation(
            invocation_id=invocation_id,
            requirement_id=requirement.requirement_id,
            capability_id=requirement.capability_id,
            provider_id=self._metadata.provider_id,
            arguments=intent.tool_arguments(),
            correlation=ToolCorrelation(
                run_id=state.run_id,
                task_id=state.task.task_id,
                action_id=action.action_id,
            ),
            requested_at=state.updated_at,
        )
        candidates = (self._metadata,)
        payload = ToolInvocationDecisionPayload(
            task_description=state.task.description,
            node_goal="Execute one bounded terminal command in the current trial.",
            invocation=invocation,
            requirement=requirement,
            provider_metadata=self._metadata,
            provider_metadata_fingerprint=tool_invocation_fingerprint(self._metadata),
            candidate_set_fingerprint=tool_invocation_candidate_set_fingerprint(
                candidates
            ),
            selection_fingerprint=tool_invocation_fingerprint(
                {
                    "provider_id": self._metadata.provider_id,
                    "trial_id": intent.trial_id,
                }
            ),
            proposal_fingerprint=tool_invocation_fingerprint(proposal),
            exact_argument_constraints={"trial_id": intent.trial_id},
            state_revision=state.revision,
            privileged=False,
        )
        basis = _invocation_basis(payload)
        request_id = stable_tool_invocation_decision_id(
            "terminal-decision-request",
            invocation_id,
            proposal.call_key,
            state.revision,
        )
        request = DecisionRequest[ToolInvocationDecisionPayload](
            request_id=request_id,
            decision_type=TOOL_INVOCATION_DECISION_TYPE,
            target=DecisionTarget(
                target_type="tool_provider",
                target_id=self._metadata.provider_id,
            ),
            correlation=DecisionCorrelation(
                run_id=state.run_id,
                task_id=state.task.task_id,
                action_id=action.action_id,
            ),
            basis=basis,
            payload=payload,
            allowed_actions=(TOOL_INVOCATION_OPERATION,),
            constraints=(
                DecisionConstraint(
                    constraint_id="terminal-trial-scope",
                    description="Invocation is bound to exactly one Harbor trial.",
                    value=intent.trial_id,
                ),
                DecisionConstraint(
                    constraint_id="terminal-explicit-shell-state",
                    description=(
                        "Command, cwd, complete env map, timeout, and process "
                        "reference are Runtime-normalized Effect fields."
                    ),
                ),
                DecisionConstraint(
                    constraint_id="terminal-no-automatic-replay",
                    description=(
                        "An unknown execution state is reconciled but never replayed."
                    ),
                ),
            ),
            budget=DecisionBudget(
                max_decision_cycles=1,
                max_agent_calls=1,
                max_revision_count=0,
                max_elapsed_seconds=(
                    intent.timeout_sec + self._policy.provider_grace_sec + 60
                ),
            ),
            created_at=state.updated_at,
        )
        projected = build_tool_invocation_agent_input(payload)
        sources = ProjectionSources(
            items=(
                ProjectionSource(
                    source_id="terminal-tool-invocation-input",
                    source_type=TOOL_INVOCATION_INPUT_SOURCE_TYPE,
                    agent_scope="tool_invocation",
                    content=projected.model_dump(mode="json"),
                    sensitivity=ContextSensitivity.INTERNAL,
                    priority=100,
                    estimated_tokens=512,
                ),
            )
        )
        context_policy = ContextProjectionPolicy(
            policy_id="terminal_bench.tool_invocation.context",
            version="1",
            agent_scope="tool_invocation",
            allowed_decision_types=frozenset({TOOL_INVOCATION_DECISION_TYPE}),
            allowed_source_types=frozenset({TOOL_INVOCATION_INPUT_SOURCE_TYPE}),
            allowed_sensitivity_levels=frozenset({ContextSensitivity.INTERNAL}),
            redact_keys=frozenset(
                {"api_key", "authorization", "credential", "password", "secret", "token"}
            ),
            max_items=1,
            max_context_tokens=768,
        )

        async def apply_effect(effect: ToolInvocationEffect) -> JsonValue:
            del effect
            raise RuntimeError("terminal Tool execution requires a Runtime Permit")

        async def apply_authorized_effect(
            normalized: NormalizedDecisionEffect[ToolInvocationEffect],
            permit: RuntimeCommitPermit,
        ) -> JsonValue:
            effect = normalized.payload
            observation = await self._executor.execute(
                effect.invocation,
                ToolExecutionPolicy(
                    timeout_seconds=(
                        intent.timeout_sec + self._policy.provider_grace_sec
                    ),
                    retry=RetryPolicy(
                        max_retries=0,
                        retry_failures=False,
                        retry_timeouts=False,
                    ),
                ),
                permit=permit,
                target=self._governance_target(normalized),
                subject_fingerprint=governance_fingerprint(normalized),
            )
            return cast(JsonValue, observation.model_dump(mode="json"))

        async def reconcile_effect(
            normalized: NormalizedDecisionEffect[ToolInvocationEffect],
        ) -> DecisionReconciliation:
            recorded = self._journal.execution_for(
                normalized.payload.invocation.invocation_id
            )
            if recorded is None:
                return DecisionReconciliation(
                    status=DecisionReconciliationStatus.UNKNOWN,
                    reason=(
                        "terminal execution has no authoritative completion record; "
                        "the command is not replayed"
                    ),
                )
            observation = terminal_tool_observation_from_result(
                normalized.payload.invocation,
                recorded,
            )
            receipt = DecisionApplyReceipt(
                effect_fingerprint=normalized.effect_fingerprint,
                committed_state_fingerprint=governance_fingerprint(
                    observation.model_dump(mode="json")
                ),
                result=observation.model_dump(mode="json"),
                applied_at=recorded.completed_at,
            )
            return DecisionReconciliation(
                status=DecisionReconciliationStatus.COMMITTED,
                reason="terminal execution journal contains the original result",
                apply_receipt=receipt,
            )

        coordinator = DecisionLifecycleCoordinator[
            ToolInvocationDecisionPayload,
            ToolInvocationProposalDraft,
            ToolInvocationEffect,
            DecisionGovernanceBinding,
        ](
            context_builder=PolicyAgentContextBuilder(),
            proposal_producer=BoundToolInvocationProposalProducer(
                draft=proposal,
                producer_id=producer_id,
            ),
            basis_provider=_TerminalInvocationBasisProvider(basis),
            validator=RuntimeDecisionValidator(
                normalizer=ToolInvocationEffectNormalizer()
            ),
            governance=RuntimeDecisionGovernanceAdapter(
                evaluator=self._governance,
                authorization_issuer=self._issuer,
                review_service=self._reviews,
            ),
            applier=GovernedDecisionApplier(
                executor=self._operation_executor,
                apply_effect=apply_effect,
                apply_authorized_effect=apply_authorized_effect,
                reconcile_effect=reconcile_effect,
            ),
            checkpoint_store=self._checkpoint_store,
            checkpoint_type=DecisionCheckpoint[
                ToolInvocationDecisionPayload,
                ToolInvocationProposalDraft,
                ToolInvocationEffect,
            ],
            trace_writer=RuntimeDecisionTraceWriter(self._trace_sink),
        )
        checkpoint = await coordinator.run(
            request,
            sources=sources,
            policy=context_policy,
        )
        result = checkpoint.result
        if (
            checkpoint.stage is not DecisionCheckpointStage.COMPLETED
            or result is None
            or result.status is not DecisionResultStatus.APPLIED
            or checkpoint.validated_decision is None
            or result.apply_receipt is None
        ):
            reason = (
                result.reason
                if result is not None
                else f"decision stage {checkpoint.stage.value}"
            )
            return TerminalToolDecisionResult(
                request_id=request_id,
                invocation=invocation,
                status=(result.status.value if result is not None else checkpoint.stage.value),
                effect_fingerprint=(
                    checkpoint.validated_decision.normalized_effect.effect_fingerprint
                    if checkpoint.validated_decision is not None
                    else None
                ),
                execution_in_doubt=(
                    result is not None
                    and result.reconciliation_status
                    is DecisionReconciliationStatus.UNKNOWN
                ),
                reason=reason,
            )
        effect = checkpoint.validated_decision.normalized_effect.payload
        observation = ToolObservation.model_validate(result.apply_receipt.result)
        outcome = ToolInvocationDecisionOutcome(
            request_id=request_id,
            effect=effect,
            observation=observation,
        )
        return TerminalToolDecisionResult(
            request_id=request_id,
            invocation=invocation,
            status=result.status.value,
            effect_fingerprint=(
                checkpoint.validated_decision.normalized_effect.effect_fingerprint
            ),
            outcome=outcome,
        )

    @staticmethod
    def _governance_target(
        normalized: NormalizedDecisionEffect[ToolInvocationEffect],
    ) -> GovernanceTarget:
        return GovernanceTarget(
            target_type=normalized.target.target_type,
            target_id=normalized.target.target_id,
        )
