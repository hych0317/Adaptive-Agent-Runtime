"""Application Strategies that compose Runtime public interfaces."""

from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Callable
from typing import Mapping
from uuid import UUID, uuid4

from pydantic import JsonValue, ValidationError

from adaptive_agent_runtime import AgentState, Observation
from adaptive_agent_runtime.context_memory import (
    ContextAssembly,
    ContextLayer,
    ContextLifecycleAction,
    ContextLifecycleDecision,
    ContextLifecycleManager,
    ContextLifecycleResult,
    ContextMemoryCoordinator,
    ContextMetadata,
    ContextRequirement,
    ContextSource,
    ContextUnit,
    MemoryRecallQuery,
    ResidencyPolicy,
)
from adaptive_agent_runtime.evaluation import ContextMemoryTraceAdapter, TraceBatch
from adaptive_agent_runtime.governance import (
    BoundGovernedOperation,
    ConfidenceSignals,
    ContextGovernanceAdapter,
    ContextGovernanceOperation,
    DecisionOutcome,
    GovernanceCorrelation,
    GovernanceEvidence,
    GovernanceHistory,
    GovernanceAuthorizationIssuer,
    GovernanceEvaluator,
    GovernanceRequest,
    GovernanceScope,
    GovernanceTarget,
    GovernedOperationExecutor,
    GovernedOperationError,
    GraphMutationGovernanceAdapter,
    RecoveryGovernanceAdapter,
    HumanReviewDecision,
    ImpactAssessment,
    InMemoryHumanReviewService,
    ReviewOutcome,
    RiskLevel,
    SUBJECT_FINGERPRINT_ATTRIBUTE,
    ToolGovernanceAdapter,
    governance_fingerprint,
)
from adaptive_agent_runtime.llm import (
    ActionProposalDraft,
    AgentBackendError,
    AgentExecutionPolicy,
    AutonomousAgentBackend,
    AutonomousAgentRequest,
    AutonomousAgentResult,
    BackendAvailability,
    BackendDelegatedAccess,
    ArtifactGenerationCapability,
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    EvidenceReference,
    GraphMutationProposalCapability,
    GraphMutationProposalDraft,
    GenerationRequest,
    InferenceCorrelation,
    LLMContextPackage,
    ReasoningCapability,
    ReasoningContext,
    MemoryCandidateDraft,
)
from adaptive_agent_runtime.orchestration import (
    DynamicTaskGraph,
    GraphMutation,
    IsolatedAgentExecutor,
    NodeExecutionResult,
    TaskNode,
    RecoveryPlan,
    apply_recovery_plan,
)
from adaptive_agent_runtime.tool_ecosystem import (
    CapabilityCandidateResolver,
    CapabilityRequest,
    CapabilityRequirement,
    ToolExecutionPolicy,
    ToolExecutionStrategy,
    ToolExecutor,
    ToolEcosystemError,
    ToolIntegrationError,
    ToolInvocation,
    ToolInvocationDecisionHandler,
    ToolInvocationProposalDraft,
    ToolObservation,
    ToolSelector,
    ToolResultObservationAdapter,
    ToolSelectionDecisionHandler,
    ToolSelectionContext,
)

from applications.research_agent.capabilities import (
    CALCULATION,
    PRIVILEGED_INFORMATION_PROVIDERS,
    DOCUMENT_ANALYSIS,
    INFORMATION_RETRIEVAL,
    REPORT_GENERATION,
)
from applications.research_agent.prompts import CHINESE_OUTPUT_INSTRUCTION
from applications.research_agent.cognition import (
    ResearchReportContextProjection,
)
from applications.research_agent.report import (
    GovernanceRecord,
    ResearchReasoningRecord,
    ResearchReport,
    ResearchToolIntentRecord,
)
from applications.research_agent.tasks import (
    COMPANY_RESEARCH,
    COMPETITOR_ANALYSIS,
    FINANCIAL_DOCUMENT,
    FINANCIAL_METRICS,
    INDUSTRY_ANALYSIS,
    NEWS_ANALYSIS,
    REPORT_GENERATION as REPORT_NODE,
    RISK_REVIEW,
    ResearchTaskDefinition,
)


def _json_dump(value: JsonValue) -> JsonValue:
    """Return JSON-compatible data without exposing mutable Runtime state."""

    return value


class ResearchWorkspace:
    """Run-scoped Application state, separate from AgentState and TaskGraph."""

    def __init__(self, definition: ResearchTaskDefinition) -> None:
        self.definition = definition
        self.outputs: dict[UUID, JsonValue] = {}
        self.context_units: list[ContextUnit] = []
        self.context_assemblies: list[ContextAssembly] = []
        self.trace_batches: list[TraceBatch] = []
        self.tool_observations: list[ToolObservation] = []
        self.governance_records: list[GovernanceRecord] = []
        self.isolated_node_ids: list[UUID] = []
        self.agent_executions: list[AutonomousAgentResult] = []
        self.reasoning_records: list[ResearchReasoningRecord] = []
        self.memory_candidate_drafts: list[MemoryCandidateDraft] = []
        self.llm_context_packages: list[LLMContextPackage] = []
        self.llm_tool_intents: list[ResearchToolIntentRecord] = []
        self.action_proposals: list[ActionProposalDraft] = []
        self.graph_mutation_proposals: list[GraphMutationProposalDraft] = []
        self.fallback_node_ids: set[UUID] = set()
        self._roles_by_node = {
            node.node_id: role for role, node in definition.nodes.items()
        }

    def role_for(self, node: TaskNode) -> str:
        return self._roles_by_node[node.node_id]

    def register_recovery_node(
        self,
        recovery_node: TaskNode,
        failed_node: TaskNode,
    ) -> None:
        """Give a Runtime-created recovery node the failed node's domain role."""

        if recovery_node.node_id in self._roles_by_node:
            raise ValueError("recovery node is already registered")
        self._roles_by_node[recovery_node.node_id] = self.role_for(failed_node)

    def set_output(self, node: TaskNode, output: JsonValue) -> None:
        self.outputs[node.node_id] = output

    def context_for(self, node_id: UUID) -> ContextAssembly | None:
        for assembly in reversed(self.context_assemblies):
            if assembly.requirement.node_id == node_id:
                return assembly
        return None

    def output_for_role(self, role: str) -> JsonValue:
        return self.outputs.get(self.definition.node(role).node_id)

    def report_preferences(self, node_id: UUID) -> tuple[JsonValue, ...]:
        for assembly in reversed(self.context_assemblies):
            if assembly.requirement.node_id != node_id:
                continue
            preferences: list[JsonValue] = []
            for unit in assembly.semantic_context:
                content = unit.content
                if isinstance(content, Mapping) and "content" in content:
                    preferences.append(_json_dump(content["content"]))
            return tuple(preferences)
        return ()


class ResearchCapabilityRequestProvider:
    """Translate Application node roles into Capability requirements."""

    module_id = "research_agent.capability_requests"

    def __init__(self, workspace: ResearchWorkspace) -> None:
        self._workspace = workspace

    def request_for(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> CapabilityRequest:
        role = self._workspace.role_for(node)
        company = self._workspace.definition.company
        capability_id, provider_tags = self._capability_for(role, node.node_id)
        arguments: dict[str, JsonValue] = {"company": company, "role": role}
        if role == FINANCIAL_METRICS:
            arguments["financials"] = self._workspace.output_for_role(
                FINANCIAL_DOCUMENT
            )
        if role == REPORT_NODE:
            arguments["analyses"] = {
                item: self._workspace.output_for_role(item)
                for item in (
                    FINANCIAL_METRICS,
                    INDUSTRY_ANALYSIS,
                    COMPETITOR_ANALYSIS,
                    NEWS_ANALYSIS,
                    RISK_REVIEW,
                )
            }
            arguments["preferences"] = list(
                self._workspace.report_preferences(node.node_id)
            )
        assembly = self._workspace.context_for(node.node_id)
        context_tags: tuple[str, ...] = ()
        context_facts: dict[str, JsonValue] = {}
        if assembly is not None:
            context_tags = tuple(
                dict.fromkeys(
                    tag
                    for unit in assembly.units
                    for tag in unit.metadata.tags
                )
            )
            context_facts = {
                "run_id": str(state.run_id),
                "context_unit_ids": [
                    str(unit.context_id) for unit in assembly.units
                ],
                "context_revisions": {
                    str(unit.context_id): unit.revision
                    for unit in assembly.units
                },
                "used_tokens": assembly.used_tokens,
            }
        return CapabilityRequest(
            requirement=CapabilityRequirement(
                capability_id=capability_id,
                required_provider_tags=provider_tags,
            ),
            arguments=arguments,
            selection_context=ToolSelectionContext(
                tags=tuple(dict.fromkeys(("research", role, *context_tags))),
                facts=context_facts,
            ),
        )

    def _capability_for(
        self,
        role: str,
        node_id: UUID,
    ) -> tuple[str, tuple[str, ...]]:
        if role == COMPANY_RESEARCH:
            return INFORMATION_RETRIEVAL, ("company",)
        if role == INDUSTRY_ANALYSIS:
            return INFORMATION_RETRIEVAL, ("industry",)
        if role == COMPETITOR_ANALYSIS:
            return INFORMATION_RETRIEVAL, ("competitor",)
        if role == NEWS_ANALYSIS:
            if node_id in self._workspace.fallback_node_ids:
                return INFORMATION_RETRIEVAL, ("news", "backup")
            return INFORMATION_RETRIEVAL, ("news",)
        if role == FINANCIAL_DOCUMENT:
            return DOCUMENT_ANALYSIS, ("financial_document",)
        if role == FINANCIAL_METRICS:
            return CALCULATION, ("financial_metrics",)
        if role == REPORT_NODE:
            return REPORT_GENERATION, ("report",)
        raise ToolIntegrationError(f"node role '{role}' has no Capability mapping")


class ResearchContextRequirementProvider:
    """Derive Context requirements outside TaskNode from accepted dependencies."""

    module_id = "research_agent.context_requirements"

    def __init__(
        self,
        workspace: ResearchWorkspace,
        *,
        max_units: int = 64,
        max_tokens: int = 8192,
    ) -> None:
        self._workspace = workspace
        self._max_units = max_units
        self._max_tokens = max_tokens

    def requirement_for(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> ContextRequirement:
        required_node_ids = set(node.dependencies)
        if self._workspace.role_for(node) == REPORT_NODE:
            required_node_ids.update(self._workspace.outputs)
        latest_by_node: dict[UUID, ContextUnit] = {}
        for unit in self._workspace.context_units:
            source_node_id = unit.metadata.node_id
            if (
                source_node_id is None
                or source_node_id not in required_node_ids
                or unit.metadata.source is not ContextSource.OBSERVATION
            ):
                continue
            current = latest_by_node.get(source_node_id)
            if current is None or unit.revision > current.revision:
                latest_by_node[source_node_id] = unit
        return ContextRequirement(
            run_id=state.run_id,
            task_id=state.task.task_id,
            node_id=node.node_id,
            goal=node.goal,
            preferred_tags=("research", f"node:{node.node_id}"),
            required_context_ids=tuple(
                latest_by_node[item].context_id
                for item in sorted(latest_by_node, key=str)
            ),
            max_units=self._max_units,
            max_tokens=self._max_tokens,
        )


class GovernedRecordingToolExecutor:
    """Application composition: authorize, delegate, then retain observations."""

    module_id = "research_agent.tool_executor.governed"

    def __init__(
        self,
        *,
        delegate: ToolExecutor,
        governance: GovernanceEvaluator,
        reviews: InMemoryHumanReviewService,
        issuer: GovernanceAuthorizationIssuer,
        operation_executor: GovernedOperationExecutor,
        workspace: ResearchWorkspace,
    ) -> None:
        self._delegate = delegate
        self._governance = governance
        self._reviews = reviews
        self._issuer = issuer
        self._operation_executor = operation_executor
        self._workspace = workspace
        self._adapter = ToolGovernanceAdapter()

    async def execute(
        self,
        invocation: ToolInvocation,
        policy: ToolExecutionPolicy,
    ) -> ToolObservation:
        privileged = invocation.provider_id in PRIVILEGED_INFORMATION_PROVIDERS
        request = self._adapter.to_request(
            invocation,
            privileged=privileged,
            impact_score=0.7 if privileged else 0.1,
        )
        preliminary = self._governance.evaluate(request)
        final = preliminary
        review = None
        if preliminary.outcome is DecisionOutcome.REVIEW_REQUIRED:
            review_id = preliminary.review_request_id
            if review_id is None:
                raise ToolIntegrationError("review-required decision has no review id")
            review = self._reviews.resolve(
                review_id,
                HumanReviewDecision(
                    outcome=ReviewOutcome.APPROVE,
                    reviewer_id="research-demo-reviewer",
                    rationale="Demo fixture access is bounded and read-only.",
                    decided_at=datetime.now(timezone.utc),
                ),
            )
            final = self._governance.finalize_review(request, review)
        authorization = None
        if final.outcome is DecisionOutcome.ALLOW:
            authorization = self._issuer.issue(request, final)
        self._workspace.governance_records.append(
            GovernanceRecord(
                scenario="tool",
                request=request,
                preliminary=preliminary,
                final=final,
                authorization=authorization,
                review=review,
            )
        )
        if authorization is None:
            raise ToolIntegrationError(
                f"Governance denied Tool invocation: {final.reason}"
            )
        async def apply() -> ToolObservation:
            return await self._delegate.execute(invocation, policy)

        observation = await self._operation_executor.execute(
            request=request,
            decision=final,
            authorization=authorization,
            target=BoundGovernedOperation(
                module_id="research_agent.tool_operation",
                operation=request.operation,
                target=request.target,
                subject=invocation,
                apply=apply,
            ),
        )
        self._workspace.tool_observations.append(observation)
        return observation


class ResearchGraphMutationApplier:
    """Enforce Governance at the exact Task Graph mutation point."""

    module_id = "research_agent.graph_mutation_applier"

    def __init__(
        self,
        *,
        authorize: Callable[[GovernanceRequest], GovernanceRecord],
        operation_executor: GovernedOperationExecutor,
        workspace: ResearchWorkspace,
    ) -> None:
        self._authorize = authorize
        self._operation_executor = operation_executor
        self._workspace = workspace
        self._adapter = GraphMutationGovernanceAdapter()

    async def apply(
        self,
        graph: DynamicTaskGraph,
        mutation: GraphMutation,
        *,
        state: AgentState,
        source_node_id: UUID,
    ) -> DynamicTaskGraph:
        action_id = (
            state.last_observation.action_id
            if state.last_observation is not None
            else None
        )
        evidence = (
            GovernanceEvidence(
                evidence_id=f"observation:{action_id}",
                kind="runtime.observation",
                source="runtime_core",
                reliability=1.0,
                summary="Successful node Observation proposed this mutation.",
            ),
            GovernanceEvidence(
                evidence_id=f"mutation:{mutation.mutation_id}",
                kind="orchestration.mutation_reason",
                source="execution_strategy",
                reliability=0.9,
                summary=mutation.reason or "Validated graph mutation proposal.",
            ),
        )
        request = self._adapter.to_request(
            mutation,
            run_id=state.run_id,
            task_id=state.task.task_id,
            node_id=source_node_id,
            action_id=action_id,
            evidence=evidence,
            history=GovernanceHistory(successful_similar=5),
        )
        record = self._authorize(request)
        self._workspace.governance_records.append(record)
        authorization = record.authorization
        if authorization is None:
            raise RuntimeError("Governance denied Task Graph mutation")

        async def apply_mutation() -> DynamicTaskGraph:
            return graph.apply_mutation(mutation)

        return await self._operation_executor.execute(
            request=request,
            decision=record.final,
            authorization=authorization,
            target=BoundGovernedOperation(
                module_id="research_agent.graph_mutation_operation",
                operation=request.operation,
                target=request.target,
                subject=mutation,
                apply=apply_mutation,
            ),
        )


class ResearchRecoveryPlanApplier:
    """Apply a Failure-driven Recovery Plan through Governance Enforcement."""

    module_id = "research_agent.recovery_plan_applier"

    def __init__(
        self,
        *,
        authorize: Callable[[GovernanceRequest], GovernanceRecord],
        operation_executor: GovernedOperationExecutor,
        workspace: ResearchWorkspace,
    ) -> None:
        self._authorize = authorize
        self._operation_executor = operation_executor
        self._workspace = workspace
        self._adapter = RecoveryGovernanceAdapter()

    async def apply(
        self,
        graph: DynamicTaskGraph,
        plan: RecoveryPlan,
        *,
        state: AgentState,
    ) -> DynamicTaskGraph:
        request = self._adapter.to_request(
            plan,
            run_id=state.run_id,
            task_id=state.task.task_id,
            action_id=plan.analysis.action_id,
            history=GovernanceHistory(successful_similar=5),
        )
        record = self._authorize(request)
        self._workspace.governance_records.append(record)
        authorization = record.authorization
        if authorization is None:
            raise RuntimeError("Governance denied Failure-driven Replanning")

        async def apply_plan() -> DynamicTaskGraph:
            return apply_recovery_plan(graph, plan)

        return await self._operation_executor.execute(
            request=request,
            decision=record.final,
            authorization=authorization,
            target=BoundGovernedOperation(
                module_id="research_agent.recovery_operation",
                operation=request.operation,
                target=request.target,
                subject=plan,
                apply=apply_plan,
            ),
        )


class GovernedContextLifecycleExecutor:
    """Application boundary that enforces every adaptive Context transition."""

    module_id = "research_agent.context_lifecycle.governed"

    _GOVERNANCE_OPERATIONS = {
        ContextLifecycleAction.COMPRESS: ContextGovernanceOperation.COMPRESS,
        ContextLifecycleAction.ARCHIVE: ContextGovernanceOperation.ARCHIVE,
        ContextLifecycleAction.RESTORE: ContextGovernanceOperation.RESTORE,
    }

    def __init__(
        self,
        *,
        manager: ContextLifecycleManager,
        authorize: Callable[[GovernanceRequest], GovernanceRecord],
        operation_executor: GovernedOperationExecutor,
        workspace: ResearchWorkspace,
    ) -> None:
        self._manager = manager
        self._authorize = authorize
        self._operation_executor = operation_executor
        self._workspace = workspace

    async def execute(
        self,
        decision: ContextLifecycleDecision,
    ) -> ContextLifecycleResult:
        subject = await self._manager.resolve_action_subject(decision)
        operation = self._GOVERNANCE_OPERATIONS[decision.action]
        request = ContextGovernanceAdapter().to_request(
            subject,
            operation=operation,
            run_id=decision.run_id,
            task_id=decision.task_id,
            node_id=decision.node_id,
        )
        record = self._authorize(request)
        self._workspace.governance_records.append(record)
        authorization = record.authorization
        if authorization is None:
            raise GovernedOperationError(
                f"Context lifecycle action '{decision.action.value}' was denied"
            )

        async def apply() -> ContextLifecycleResult:
            return await self._manager.apply(decision)

        return await self._operation_executor.execute(
            request=request,
            decision=record.final,
            authorization=authorization,
            target=BoundGovernedOperation(
                module_id=self.module_id,
                operation=request.operation,
                target=request.target,
                subject=subject,
                apply=apply,
            ),
        )


class _ContextAwareStrategy:
    def __init__(
        self,
        *,
        workspace: ResearchWorkspace,
        lifecycle: ContextLifecycleManager,
        coordinator: ContextMemoryCoordinator,
        context_trace: ContextMemoryTraceAdapter,
    ) -> None:
        self._workspace = workspace
        self._lifecycle = lifecycle
        self._coordinator = coordinator
        self._context_trace = context_trace
        self._tool_adapter = ToolResultObservationAdapter()

    async def _prepare(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> ContextAssembly:
        current = ContextUnit(
            content={
                "current_goal": node.goal,
                "expected_output": node.expected_output,
            },
            metadata=ContextMetadata(
                source=ContextSource.WORKING_STATE,
                layer=ContextLayer.WORKING,
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=node.node_id,
                source_reference=f"task-node:{node.node_id}",
                tags=("research", "current_task", f"node:{node.node_id}"),
                importance=0.9,
                estimated_tokens=16,
            ),
            residency_policy=ResidencyPolicy.SESSION,
        )
        await self._lifecycle.add(current)
        self._workspace.context_units.append(current)
        self._workspace.trace_batches.append(
            self._context_trace.context_unit(current, kind="context.current_task")
        )
        preparation = await self._coordinator.prepare(
            state,
            node,
            memory_query=MemoryRecallQuery(
                facts={"domain": "financial_research"},
                tags=("research",),
                min_confidence=0.5,
            ),
        )
        for lifecycle_result in preparation.lifecycle_results:
            if lifecycle_result.unit is not None:
                self._workspace.context_units.append(lifecycle_result.unit)
            self._workspace.trace_batches.append(
                self._context_trace.context_lifecycle(
                    lifecycle_result,
                    run_id=state.run_id,
                    task_id=state.task.task_id,
                    node_id=node.node_id,
                )
            )
        assembly = preparation.assembly
        self._workspace.context_assemblies.append(assembly)
        self._workspace.trace_batches.append(
            self._context_trace.context_assembly(assembly)
        )
        return assembly

    async def _record_tool_observations(
        self,
        start_index: int,
        node: TaskNode,
        state: AgentState,
    ) -> tuple[ContextUnit, ...]:
        units: list[ContextUnit] = []
        for result in self._workspace.tool_observations[start_index:]:
            observation = self._tool_adapter.convert(
                result,
                action_id=result.invocation_id,
            )
            unit = await self._coordinator.record_observation(
                observation,
                state,
                node=node,
            )
            self._workspace.context_units.append(unit)
            units.append(unit)
            self._workspace.trace_batches.append(
                self._context_trace.context_unit(
                    unit,
                    action_id=observation.action_id,
                    kind="context.tool_observation",
                )
            )
        return tuple(units)


class ResearchStrategy(_ContextAwareStrategy):
    module_id = "research_agent.strategy.research"
    strategy_id = "research"

    def __init__(
        self,
        *,
        tool_strategy: ToolExecutionStrategy,
        workspace: ResearchWorkspace,
        lifecycle: ContextLifecycleManager,
        coordinator: ContextMemoryCoordinator,
        context_trace: ContextMemoryTraceAdapter,
        reasoner: ReasoningCapability | None = None,
        mutation_planner: GraphMutationProposalCapability | None = None,
        tool_invocation_handler: ToolInvocationDecisionHandler | None = None,
        max_reasoning_tool_intents: int = 0,
    ) -> None:
        super().__init__(
            workspace=workspace,
            lifecycle=lifecycle,
            coordinator=coordinator,
            context_trace=context_trace,
        )
        self._tool_strategy = tool_strategy
        self._reasoner = reasoner
        self._mutation_planner = mutation_planner
        self._tool_invocation_handler = tool_invocation_handler
        self._max_reasoning_tool_intents = max_reasoning_tool_intents

    async def execute(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> NodeExecutionResult:
        await self._prepare(node, state)
        start_index = len(self._workspace.tool_observations)
        result = await self._tool_strategy.execute(node, state)
        role = self._workspace.role_for(node)
        if not result.succeeded and role == NEWS_ANALYSIS:
            self._workspace.fallback_node_ids.add(node.node_id)
            result = await self._tool_strategy.execute(node, state)
        recorded_units = await self._record_tool_observations(
            start_index,
            node,
            state,
        )
        if not result.succeeded:
            return result
        if self._reasoner is not None:
            evidence = [
                EvidenceReference(
                    reference_id=f"node-output:{node.node_id}",
                    kind="tool.observation",
                    summary="Runtime-accepted Tool output for this task node.",
                )
            ]
            tool_feedback: list[JsonValue] = []
            used_call_keys: set[str] = set()
            used_intents = 0
            while True:
                reasoning_turn = await self._reasoner.analyze(
                    ReasoningContext(
                    goal=(
                        f"Analyze the accepted evidence for task: {node.goal}"
                    ),
                    context={
                        "task_output": result.output,
                        "runtime_tool_observations": tool_feedback,
                    },
                    evidence=tuple(evidence),
                    constraints=(
                        CHINESE_OUTPUT_INSTRUCTION,
                        "Do not invent facts beyond the supplied Tool output.",
                        "Return analysis only; do not propose state changes.",
                        "Tool calls are proposals executed only by Runtime.",
                        (
                            "如果候选工具能显著降低关键不确定性，可优先提出一次 "
                            "Tool Intent；否则直接完成分析。"
                        ),
                    ),
                ),
                    invocation=CapabilityInvocationMetadata(
                        correlation=InferenceCorrelation(
                            run_id=state.run_id,
                            task_id=state.task.task_id,
                            node_id=node.node_id,
                            action_id=uuid4(),
                        ),
                        trace_attributes={"operation": "node.reason"},
                    ),
                )
                if reasoning_turn.kind is CapabilityTurnKind.COMPLETED:
                    if reasoning_turn.result is None:
                        return NodeExecutionResult.failed(
                            error="Reasoner returned no analysis"
                        )
                    self._workspace.reasoning_records.append(
                        ResearchReasoningRecord(
                            node_id=node.node_id,
                            role=role,
                            result=reasoning_turn.result,
                        )
                    )
                    break
                handler = self._tool_invocation_handler
                intents = reasoning_turn.tool_intents
                call_keys = {intent.call_key for intent in intents}
                if handler is None:
                    return NodeExecutionResult.failed(
                        error="Reasoner ToolIntent execution is not enabled"
                    )
                if used_call_keys.intersection(call_keys):
                    return NodeExecutionResult.failed(
                        error="Reasoner replayed a ToolIntent call key"
                    )
                if used_intents + len(intents) > self._max_reasoning_tool_intents:
                    return NodeExecutionResult.failed(
                        error="Reasoner exceeded its ToolIntent budget"
                    )
                intent_observation_start = len(self._workspace.tool_observations)
                failed_observation: ToolObservation | None = None
                try:
                    for intent in intents:
                        outcome = await handler.handle(
                            proposal=ToolInvocationProposalDraft.model_validate(
                                intent.model_dump(mode="python")
                            ),
                            producer_id=self._reasoner.capability_id,
                            node=node,
                            state=state,
                        )
                        observation = outcome.observation
                        self._workspace.llm_tool_intents.append(
                            ResearchToolIntentRecord(
                                node_id=node.node_id,
                                intent=intent,
                                observation=observation,
                            )
                        )
                        reference_id = f"tool-intent:{intent.call_key}"
                        evidence.append(
                            EvidenceReference(
                                reference_id=reference_id,
                                kind="tool.observation",
                                summary=(
                                    "Runtime-governed ToolIntent observation: "
                                    f"{observation.status.value}."
                                ),
                            )
                        )
                        tool_feedback.append(
                            {
                                "evidence_reference_id": reference_id,
                                "observation": observation.model_dump(mode="json"),
                            }
                        )
                        if not observation.succeeded:
                            failed_observation = observation
                            break
                except ToolEcosystemError as exc:
                    return NodeExecutionResult.failed(error=str(exc))
                recorded_units += await self._record_tool_observations(
                    intent_observation_start,
                    node,
                    state,
                )
                if failed_observation is not None:
                    return NodeExecutionResult.failed(
                        error=(
                            failed_observation.error
                            or "Reasoner Tool invocation failed"
                        )
                    )
                used_call_keys.update(call_keys)
                used_intents += len(intents)
        self._workspace.set_output(node, result.output)
        mutations = (
            await self._company_discovery_mutations(
                node,
                state,
                result.output,
                recorded_units,
            )
            if role == COMPANY_RESEARCH
            else ()
        )
        return NodeExecutionResult.ok(
            output=result.output,
            mutations=mutations,
        )

    async def _company_discovery_mutations(
        self,
        node: TaskNode,
        state: AgentState,
        output: JsonValue,
        recorded_units: tuple[ContextUnit, ...],
    ) -> tuple[GraphMutation, ...]:
        planner = self._mutation_planner
        if planner is None:
            return self._workspace.definition.discovery_mutations()
        # The Agent-backed path is handled after Observation consumption by
        # GraphMutationDecisionHandler. Returning no metadata mutation prevents
        # the proposal from bypassing unified validation, Governance, and Trace.
        del node, state, output, recorded_units
        return ()


class DeterministicRiskAgent:
    """A selectively isolated reviewer, not a collaborating Agent swarm."""

    module_id = "research_agent.isolated.risk"

    def __init__(self, workspace: ResearchWorkspace) -> None:
        self._workspace = workspace

    async def execute_isolated(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> NodeExecutionResult:
        del state
        metrics = self._workspace.output_for_role(FINANCIAL_METRICS)
        industry = self._workspace.output_for_role(INDUSTRY_ANALYSIS)
        return NodeExecutionResult.ok(
            output={
                "reviewer": "isolated-risk-agent",
                "independent": True,
                "node_id": str(node.node_id),
                "risks": [
                    "Demand and pricing pressure may compress margins.",
                    "Execution delays could weaken the growth thesis.",
                    "Regulatory or competitive changes may alter industry economics.",
                ],
                "evidence_reviewed": {
                    "financial_metrics": metrics is not None,
                    "industry_analysis": industry is not None,
                },
            }
        )


class LLMRiskAgent:
    """Governed adapter from autonomous Agent Backend to orchestration result."""

    module_id = "research_agent.isolated.llm_risk"

    def __init__(
        self,
        *,
        backend: AutonomousAgentBackend,
        workspace: ResearchWorkspace,
        isolated_workspace_root: str,
        authorize: Callable[[GovernanceRequest], GovernanceRecord],
        operation_executor: GovernedOperationExecutor,
    ) -> None:
        self._backend = backend
        self._workspace = workspace
        self._isolated_workspace_root = isolated_workspace_root
        self._authorize = authorize
        self._operation_executor = operation_executor

    async def execute_isolated(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> NodeExecutionResult:
        request = AutonomousAgentRequest(
            goal=f"{node.goal}\n{CHINESE_OUTPUT_INSTRUCTION}",
            input={
                "company": self._workspace.definition.company,
                "node_id": str(node.node_id),
                "financial_metrics": self._workspace.output_for_role(
                    FINANCIAL_METRICS
                ),
                "industry_analysis": self._workspace.output_for_role(
                    INDUSTRY_ANALYSIS
                ),
            },
            workspace_root=self._isolated_workspace_root,
            response_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reviewer": {"type": "string", "minLength": 1},
                    "independent": {"const": True},
                    "node_id": {"type": "string", "minLength": 1},
                    "risks": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "evidence_reviewed": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "financial_metrics": {"type": "boolean"},
                            "industry_analysis": {"type": "boolean"},
                        },
                        "required": [
                            "financial_metrics",
                            "industry_analysis",
                        ],
                    },
                },
                "required": [
                    "reviewer",
                    "independent",
                    "node_id",
                    "risks",
                    "evidence_reviewed",
                ],
            },
            policy=AgentExecutionPolicy(
                delegated_access=BackendDelegatedAccess(
                    filesystem_read=True,
                    shell_execution=True,
                ),
                max_action_events=16,
                max_capture_characters=250_000,
            ),
            correlation=InferenceCorrelation(
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=node.node_id,
            ),
            timeout_seconds=120.0,
            trace_attributes={"application": "research_agent", "role": RISK_REVIEW},
        )
        governance_request = GovernanceRequest(
            scope=GovernanceScope.ACTION,
            operation="agent.execute",
            target=GovernanceTarget(
                target_type="agent_backend",
                target_id=self._backend.target_id,
            ),
            risk=RiskLevel.MEDIUM,
            signals=ConfidenceSignals(
                stated_confidence=0.9,
                evidence=(
                    GovernanceEvidence(
                        evidence_id=f"node:{node.node_id}",
                        kind="task_node",
                        source="research_task_graph",
                        reliability=1.0,
                        summary="Runtime selected the isolated risk-review node.",
                    ),
                    GovernanceEvidence(
                        evidence_id=f"workspace:{request.request_id}",
                        kind="isolation_policy",
                        source="research_agent",
                        reliability=0.95,
                        summary=(
                            "Execution uses an ephemeral read-only workspace "
                            "without network or MCP access."
                        ),
                    ),
                ),
                impact=ImpactAssessment(
                    score=0.2,
                    reversible=True,
                    description=(
                        "The Agent produces a review draft and cannot apply it."
                    ),
                ),
                history=GovernanceHistory(successful_similar=5),
            ),
            correlation=GovernanceCorrelation(
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=node.node_id,
                action_id=request.request_id,
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(request),
                "agent_request_id": str(request.request_id),
                "filesystem_write": False,
                "network": False,
                "mcp": False,
            },
        )
        governance_record = self._authorize(governance_request)
        self._workspace.governance_records.append(governance_record)
        if governance_record.authorization is None:
            return NodeExecutionResult.failed(
                error="Governance denied isolated Agent execution"
            )
        probe = await self._backend.probe()
        if probe.availability is not BackendAvailability.AVAILABLE:
            return NodeExecutionResult.failed(
                error=f"Agent backend is {probe.availability.value}"
            )
        async def execute_agent() -> AutonomousAgentResult:
            return await self._backend.execute(request)

        try:
            result = await self._operation_executor.execute(
                request=governance_request,
                decision=governance_record.final,
                authorization=governance_record.authorization,
                target=BoundGovernedOperation(
                    module_id="research_agent.isolated_agent_operation",
                    operation=governance_request.operation,
                    target=governance_request.target,
                    subject=request,
                    apply=execute_agent,
                ),
            )
        except (AgentBackendError, GovernedOperationError) as exc:
            return NodeExecutionResult.failed(
                error=f"Agent backend failed: {type(exc).__name__}"
            )
        self._workspace.agent_executions.append(result)
        output = result.model_dump(mode="json")["output"]
        return NodeExecutionResult.ok(output=output)


class ReviewStrategy(_ContextAwareStrategy):
    module_id = "research_agent.strategy.review"
    strategy_id = "review"

    def __init__(
        self,
        *,
        isolated_executor: IsolatedAgentExecutor,
        workspace: ResearchWorkspace,
        lifecycle: ContextLifecycleManager,
        coordinator: ContextMemoryCoordinator,
        context_trace: ContextMemoryTraceAdapter,
    ) -> None:
        super().__init__(
            workspace=workspace,
            lifecycle=lifecycle,
            coordinator=coordinator,
            context_trace=context_trace,
        )
        self._isolated_executor = isolated_executor

    async def execute(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> NodeExecutionResult:
        await self._prepare(node, state)
        result = await self._isolated_executor.execute_isolated(node, state)
        if not result.succeeded:
            return result
        self._workspace.isolated_node_ids.append(node.node_id)
        self._workspace.set_output(node, result.output)
        observation = Observation.ok(uuid4(), output=result.output)
        unit = await self._coordinator.record_observation(
            observation,
            state,
            node=node,
        )
        self._workspace.context_units.append(unit)
        self._workspace.trace_batches.append(
            self._context_trace.context_unit(
                unit,
                action_id=observation.action_id,
                kind="context.isolated_agent_result",
            )
        )
        return result


class ReportStrategy(_ContextAwareStrategy):
    module_id = "research_agent.strategy.report"
    strategy_id = "report"

    def __init__(
        self,
        *,
        tool_strategy: ToolExecutionStrategy,
        workspace: ResearchWorkspace,
        lifecycle: ContextLifecycleManager,
        coordinator: ContextMemoryCoordinator,
        context_trace: ContextMemoryTraceAdapter,
        llm_generator: ArtifactGenerationCapability | None = None,
        llm_context: ResearchReportContextProjection | None = None,
    ) -> None:
        super().__init__(
            workspace=workspace,
            lifecycle=lifecycle,
            coordinator=coordinator,
            context_trace=context_trace,
        )
        self._tool_strategy = tool_strategy
        self._llm_generator = llm_generator
        self._llm_context = llm_context

    async def execute(
        self,
        node: TaskNode,
        state: AgentState,
    ) -> NodeExecutionResult:
        assembly = await self._prepare(node, state)
        llm_generator = self._llm_generator
        if llm_generator is not None:
            return await self._generate_with_llm(
                node,
                llm_generator,
                assembly,
            )
        start_index = len(self._workspace.tool_observations)
        result = await self._tool_strategy.execute(node, state)
        await self._record_tool_observations(start_index, node, state)
        if result.succeeded:
            self._workspace.set_output(node, result.output)
        return result

    async def _generate_with_llm(
        self,
        node: TaskNode,
        generator: ArtifactGenerationCapability,
        assembly: ContextAssembly,
    ) -> NodeExecutionResult:
        roles = (
            FINANCIAL_METRICS,
            INDUSTRY_ANALYSIS,
            COMPETITOR_ANALYSIS,
            NEWS_ANALYSIS,
            RISK_REVIEW,
        )
        analyses = {
            role: self._workspace.output_for_role(role) for role in roles
        }
        references = tuple(
            EvidenceReference(
                reference_id=f"analysis:{role}",
                kind="research.analysis",
                summary=f"Runtime-accepted output for {role}.",
            )
            for role in roles
            if analyses[role] is not None
        )
        context: dict[str, JsonValue] = {
            "company": self._workspace.definition.company,
            "analyses": analyses,
        }
        if self._llm_context is not None:
            package = self._llm_context.project(assembly)
            self._workspace.llm_context_packages.append(package)
            context["runtime_context"] = package.model_dump(mode="json")
        request = GenerationRequest(
            instruction=(
                "Generate the final structured investment research report. "
                "Use only supplied analyses and cite their evidence reference IDs. "
                + CHINESE_OUTPUT_INSTRUCTION
            ),
            context=context,
            media_type="application/json",
            output_schema=ResearchReport.model_json_schema(mode="validation"),
            evidence=references,
        )
        turn = await generator.generate(
            request,
            invocation=CapabilityInvocationMetadata(
                correlation=InferenceCorrelation(
                    run_id=assembly.requirement.run_id,
                    task_id=assembly.requirement.task_id,
                    node_id=assembly.requirement.node_id,
                    action_id=uuid4(),
                ),
                trace_attributes={"operation": "report.generate"},
            ),
        )
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            return NodeExecutionResult.failed(
                error="Report Generator returned ToolIntent to Runtime"
            )
        if turn.result is None:
            return NodeExecutionResult.failed(
                error="Report Generator returned no artifact"
            )
        if turn.result.media_type != "application/json":
            return NodeExecutionResult.failed(
                error="Report Generator returned the wrong media type"
            )
        try:
            report = ResearchReport.model_validate(turn.result.content)
        except ValidationError:
            return NodeExecutionResult.failed(
                error="Report Generator returned an invalid report"
            )
        output = report.model_dump(mode="json")
        self._workspace.set_output(node, output)
        return NodeExecutionResult.ok(output=output)


def build_tool_execution_strategy(
    *,
    workspace: ResearchWorkspace,
    executor: ToolExecutor,
    resolver: CapabilityCandidateResolver,
    selector: ToolSelector | None = None,
    selection_handler: ToolSelectionDecisionHandler | None = None,
    timeout_seconds: float = 2.0,
) -> ToolExecutionStrategy:
    """Small typed seam kept here so Strategies reuse the Runtime adapter."""

    return ToolExecutionStrategy(
        requests=ResearchCapabilityRequestProvider(workspace),
        resolver=resolver,
        selector=selector,
        selection_handler=selection_handler,
        executor=executor,
        policy=ToolExecutionPolicy(timeout_seconds=timeout_seconds),
        strategy_id="research-tool-delegate",
    )
