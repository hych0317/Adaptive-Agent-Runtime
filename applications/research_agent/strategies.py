"""Application Strategies that compose Runtime public interfaces."""

from __future__ import annotations

from datetime import datetime, timezone
from collections.abc import Callable
from typing import Mapping
from uuid import UUID, uuid4

from pydantic import JsonValue, ValidationError
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from adaptive_agent_runtime import AgentState, Observation
from adaptive_agent_runtime.context_memory import (
    ContextAssembly,
    ContextCompressionResult,
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
    ActionProposalCapability,
    ActionProposalDraft,
    ActionProposalRequest,
    AgentBackendError,
    AgentExecutionPolicy,
    AutonomousAgentBackend,
    AutonomousAgentRequest,
    AutonomousAgentResult,
    BackendAvailability,
    BackendDelegatedAccess,
    ArtifactGenerationCapability,
    CapabilityInvocationMetadata,
    CapabilityDraftValidator,
    CapabilityTurnKind,
    CompressionRequest,
    EvidenceReference,
    GraphMutationProposalCapability,
    GraphMutationProposalDraft,
    GraphMutationProposalRequest,
    GenerationRequest,
    InferenceCorrelation,
    LLMContextPackage,
    PlanningActionCandidate,
    ReasoningCapability,
    ReasoningContext,
    SemanticCompressionCapability,
    MemoryCandidateDraft,
    ToolIntentDraft,
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
    ToolCorrelation,
    ToolObservation,
    ToolResultObservationAdapter,
    ToolSelector,
    ToolSelectionContext,
)

from applications.research_agent.capabilities import (
    CALCULATION,
    COMPANY_PROVIDER,
    DOCUMENT_ANALYSIS,
    INFORMATION_RETRIEVAL,
    REPORT_GENERATION,
)
from applications.research_agent.cognition import (
    ResearchContextProjection,
    ResearchReportContextProjection,
)
from applications.research_agent.report import (
    GovernanceRecord,
    ResearchReasoningRecord,
    ResearchReport,
    ResearchToolIntentRecord,
)
from applications.research_agent.llm_tools import (
    RESEARCH_INFORMATION_RETRIEVAL_TOOL,
    RESEARCH_RETRIEVAL_SCOPES,
)
from applications.research_agent.tasks import (
    COMPANY_RESEARCH,
    COMPETITOR_ANALYSIS,
    FINANCIAL_DOCUMENT,
    FINANCIAL_METRICS,
    INDUSTRY_ANALYSIS,
    NEWS_ANALYSIS,
    RESEARCH_NODE_ROLES,
    RESEARCH_STRATEGY_ID,
    REVIEW_STRATEGY_ID,
    REPORT_GENERATION as REPORT_NODE,
    RISK_REVIEW,
    ResearchTaskDefinition,
    build_research_mutations_from_draft,
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


class LLMReadyTaskNodeSelector:
    """Turn an LLM action draft into a governed ready-node selection."""

    module_id = "research_agent.ready_node_selector.llm"

    def __init__(
        self,
        *,
        capability: ActionProposalCapability,
        workspace: ResearchWorkspace,
        authorize: Callable[[GovernanceRequest], GovernanceRecord],
        operation_executor: GovernedOperationExecutor,
    ) -> None:
        self._capability = capability
        self._workspace = workspace
        self._authorize = authorize
        self._operation_executor = operation_executor
        self._validator = CapabilityDraftValidator()

    async def select_node_id(
        self,
        ready_nodes: tuple[TaskNode, ...],
        state: AgentState,
    ) -> UUID:
        if not ready_nodes:
            raise RuntimeError("LLM action selection requires ready nodes")
        if len(ready_nodes) == 1:
            return ready_nodes[0].node_id
        candidates = tuple(
            PlanningActionCandidate(
                node_key=self._workspace.role_for(node),
                goal=node.goal,
                expected_output=node.expected_output,
                strategy_id=node.strategy_id,
            )
            for node in ready_nodes
        )
        evidence_id = f"runtime:ready:{state.run_id}:{state.step_count}"
        request = ActionProposalRequest(
            task=state.task.description,
            candidates=candidates,
            evidence=(
                EvidenceReference(
                    reference_id=evidence_id,
                    kind="runtime.ready_set",
                    summary="Runtime scheduler produced the candidate set.",
                    reliability=1.0,
                ),
            ),
        )
        proposal_action_id = uuid4()
        turn = await self._capability.propose_action(
            request,
            invocation=CapabilityInvocationMetadata(
                correlation=InferenceCorrelation(
                    run_id=state.run_id,
                    task_id=state.task.task_id,
                    action_id=proposal_action_id,
                ),
                trace_attributes={"operation": "node.select.propose"},
            ),
        )
        if turn.kind is CapabilityTurnKind.TOOL_INTENT or turn.result is None:
            raise RuntimeError("Action Planner returned no action proposal")
        proposal = self._validator.validate_action_proposal(
            request,
            turn.result,
        )
        by_key = {
            self._workspace.role_for(node): node for node in ready_nodes
        }
        selected = by_key.get(proposal.node_key)
        if selected is None:
            raise RuntimeError("Action Planner escaped the Runtime ready set")
        governance_request = GovernanceRequest(
            scope=GovernanceScope.ACTION,
            operation="node.select",
            target=GovernanceTarget(
                target_type="task_node",
                target_id=str(selected.node_id),
            ),
            risk=RiskLevel.LOW,
            signals=ConfidenceSignals(
                stated_confidence=0.9,
                evidence=(
                    GovernanceEvidence(
                        evidence_id=evidence_id,
                        kind="runtime.ready_set",
                        source="graph_scheduler",
                        reliability=1.0,
                        summary=(
                            "The proposed node belongs to the Runtime ready set."
                        ),
                    ),
                ),
                impact=ImpactAssessment(
                    score=0.1,
                    reversible=True,
                    description=(
                        "The proposal orders one already-ready node for this run."
                    ),
                ),
                history=GovernanceHistory(successful_similar=5),
            ),
            correlation=GovernanceCorrelation(
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=selected.node_id,
                action_id=proposal_action_id,
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(proposal),
                "candidate_node_keys": [
                    candidate.node_key for candidate in candidates
                ],
                "selected_node_key": proposal.node_key,
                "planner_capability_id": self._capability.capability_id,
            },
        )
        record = self._authorize(governance_request)
        self._workspace.action_proposals.append(proposal)
        self._workspace.governance_records.append(record)
        if record.authorization is None:
            raise RuntimeError("Governance denied LLM Action Proposal")
        async def select() -> UUID:
            return selected.node_id

        return await self._operation_executor.execute(
            request=governance_request,
            decision=record.final,
            authorization=record.authorization,
            target=BoundGovernedOperation(
                module_id="research_agent.node_selection",
                operation=governance_request.operation,
                target=governance_request.target,
                subject=proposal,
                apply=select,
            ),
        )


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
        privileged = invocation.provider_id == COMPANY_PROVIDER
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


class DeterministicContextCompressor:
    """Application compressor used to demonstrate the public lifecycle contract."""

    module_id = "research_agent.context_compressor.deterministic"

    async def compress(self, unit: ContextUnit) -> ContextCompressionResult:
        return ContextCompressionResult(
            content={
                "summary": "Research evidence retained as a compact conclusion.",
                "source_reference": unit.metadata.source_reference,
            },
            core_conclusions=(
                "The evidence was consumed by the current research run.",
            ),
            estimated_tokens=min(unit.metadata.estimated_tokens, 12),
        )


class LLMContextCompressor:
    """Translate a semantic Compression draft into Context lifecycle output."""

    module_id = "research_agent.context_compressor.llm"

    def __init__(
        self,
        capability: SemanticCompressionCapability,
        context_projection: ResearchContextProjection,
    ) -> None:
        self._capability = capability
        self._context_projection = context_projection
        self._packages_by_run: dict[UUID, list[LLMContextPackage]] = {}

    def take_packages(self, run_id: UUID) -> tuple[LLMContextPackage, ...]:
        return tuple(self._packages_by_run.pop(run_id, ()))

    async def compress(self, unit: ContextUnit) -> ContextCompressionResult:
        original_tokens = unit.metadata.estimated_tokens
        if original_tokens <= 1:
            raise RuntimeError("Context Unit is too small for semantic compression")
        target_tokens = max(1, original_tokens // 2)
        layer_buckets = {
            ContextLayer.WORKING: (unit,),
            ContextLayer.TASK: (unit,),
            ContextLayer.SEMANTIC: (unit,),
        }
        assembly = ContextAssembly(
            requirement=ContextRequirement(
                run_id=unit.metadata.run_id,
                task_id=unit.metadata.task_id,
                node_id=unit.metadata.node_id,
                goal="Compress one policy-approved Runtime Context Unit.",
                layers=(unit.metadata.layer,),
                required_context_ids=(unit.context_id,),
                max_units=1,
                max_tokens=original_tokens,
            ),
            units=(unit,),
            working_context=(
                layer_buckets[ContextLayer.WORKING]
                if unit.metadata.layer is ContextLayer.WORKING
                else ()
            ),
            task_context=(
                layer_buckets[ContextLayer.TASK]
                if unit.metadata.layer is ContextLayer.TASK
                else ()
            ),
            semantic_context=(
                layer_buckets[ContextLayer.SEMANTIC]
                if unit.metadata.layer is ContextLayer.SEMANTIC
                else ()
            ),
            used_tokens=original_tokens,
        )
        package = self._context_projection.project(assembly)
        self._packages_by_run.setdefault(unit.metadata.run_id, []).append(package)
        turn = await self._capability.compress(
            CompressionRequest(
                source_reference_id=str(unit.context_id),
                content=package.model_dump(mode="json"),
                original_estimated_tokens=original_tokens,
                target_max_tokens=target_tokens,
            ),
            invocation=CapabilityInvocationMetadata(
                correlation=InferenceCorrelation(
                    run_id=unit.metadata.run_id,
                    task_id=unit.metadata.task_id,
                    node_id=unit.metadata.node_id,
                    action_id=uuid4(),
                ),
                trace_attributes={"operation": "context.compress"},
            ),
        )
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Context Compressor returned ToolIntent to Runtime")
        if turn.result is None:
            raise RuntimeError("Context Compressor returned no draft")
        return ContextCompressionResult(
            content=turn.result.content,
            core_conclusions=turn.result.core_conclusions,
            estimated_tokens=turn.result.estimated_tokens,
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


class GovernedToolIntentExecutor:
    """Validate and execute an LLM Tool proposal through Runtime authority."""

    module_id = "research_agent.tool_intent_executor.governed"

    def __init__(
        self,
        *,
        resolver: CapabilityCandidateResolver,
        selector: ToolSelector,
        executor: ToolExecutor,
        workspace: ResearchWorkspace,
    ) -> None:
        self._resolver = resolver
        self._selector = selector
        self._executor = executor
        self._workspace = workspace
        self._policy = ToolExecutionPolicy(timeout_seconds=2.0)
        schema = RESEARCH_INFORMATION_RETRIEVAL_TOOL.model_dump(mode="json")[
            "input_schema"
        ]
        self._validator = Draft202012Validator(schema)

    async def execute(
        self,
        intent: ToolIntentDraft,
        node: TaskNode,
        state: AgentState,
    ) -> ToolObservation:
        if intent.capability_id != RESEARCH_INFORMATION_RETRIEVAL_TOOL.capability_id:
            raise ToolIntegrationError(
                f"ToolIntent capability '{intent.capability_id}' is not allowed"
            )
        arguments = intent.model_dump(mode="json")["arguments"]
        errors = tuple(self._validator.iter_errors(arguments))
        if errors:
            raise ToolIntegrationError(
                "ToolIntent arguments do not match the Runtime schema"
            )
        if arguments["company"] != self._workspace.definition.company:
            raise ToolIntegrationError(
                "ToolIntent cannot change the company task boundary"
            )
        scope = str(arguments["scope"])
        if scope not in RESEARCH_RETRIEVAL_SCOPES:
            raise ToolIntegrationError("ToolIntent retrieval scope is not allowed")
        requirement = CapabilityRequirement(
            capability_id=intent.capability_id,
            required_provider_tags=(scope,),
        )
        candidates = self._resolver.candidates(requirement)
        selection = self._selector.select(
            requirement,
            candidates,
            ToolSelectionContext(tags=("llm_tool_intent", scope)),
        )
        if selection.requirement_id != requirement.requirement_id:
            raise ToolIntegrationError(
                "Tool selector returned a mismatched requirement"
            )
        if selection.capability_id != requirement.capability_id:
            raise ToolIntegrationError("Tool selector changed the capability")
        if selection.provider_id not in {
            candidate.provider_id for candidate in candidates
        }:
            raise ToolIntegrationError(
                "Tool selector escaped Runtime-filtered candidates"
            )
        invocation_id = uuid4()
        invocation = ToolInvocation(
            invocation_id=invocation_id,
            requirement_id=requirement.requirement_id,
            capability_id=requirement.capability_id,
            provider_id=selection.provider_id,
            arguments=arguments,
            correlation=ToolCorrelation(
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=node.node_id,
                action_id=invocation_id,
            ),
        )
        observation = await self._executor.execute(invocation, self._policy)
        self._workspace.llm_tool_intents.append(
            ResearchToolIntentRecord(
                node_id=node.node_id,
                intent=intent,
                observation=observation,
            )
        )
        return observation


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
        mutation_context: ResearchContextProjection | None = None,
        authorize_mutation: Callable[[GovernanceRequest], GovernanceRecord]
        | None = None,
        tool_intent_executor: GovernedToolIntentExecutor | None = None,
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
        self._mutation_context = mutation_context
        self._authorize_mutation = authorize_mutation
        self._draft_validator = CapabilityDraftValidator()
        self._tool_intent_executor = tool_intent_executor
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
                        "Do not invent facts beyond the supplied Tool output.",
                        "Return analysis only; do not propose state changes.",
                        "Tool calls are proposals executed only by Runtime.",
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
                executor = self._tool_intent_executor
                intents = reasoning_turn.tool_intents
                call_keys = {intent.call_key for intent in intents}
                if executor is None:
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
                try:
                    for intent in intents:
                        observation = await executor.execute(intent, node, state)
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
                except ToolEcosystemError as exc:
                    return NodeExecutionResult.failed(error=str(exc))
                recorded_units += await self._record_tool_observations(
                    intent_observation_start,
                    node,
                    state,
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
        projection = self._mutation_context
        authorize = self._authorize_mutation
        if projection is None or authorize is None:
            raise RuntimeError(
                "Mutation Planner requires Context projection and Governance"
            )
        if not recorded_units:
            raise RuntimeError("Mutation Planner has no accepted observation")
        working = tuple(
            unit
            for unit in recorded_units
            if unit.metadata.layer is ContextLayer.WORKING
        )
        task_units = tuple(
            unit
            for unit in recorded_units
            if unit.metadata.layer is ContextLayer.TASK
        )
        semantic = tuple(
            unit
            for unit in recorded_units
            if unit.metadata.layer is ContextLayer.SEMANTIC
        )
        used_tokens = sum(
            unit.metadata.estimated_tokens for unit in recorded_units
        )
        ordered_units = (*working, *task_units, *semantic)
        assembly = ContextAssembly(
            requirement=ContextRequirement(
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=node.node_id,
                goal="Propose bounded graph mutations from company evidence.",
                max_units=len(recorded_units),
                max_tokens=used_tokens,
            ),
            units=ordered_units,
            working_context=working,
            task_context=task_units,
            semantic_context=semantic,
            used_tokens=used_tokens,
        )
        package = projection.project(assembly)
        if not package.blocks:
            raise RuntimeError(
                "Mutation Planner Context policy omitted all observations"
            )
        self._workspace.llm_context_packages.append(package)
        evidence = tuple(
            EvidenceReference(
                reference_id=f"context:{block.context_id}",
                kind="tool.observation",
                summary="Policy-approved company research observation.",
                reliability=1.0,
            )
            for block in package.blocks
        )
        existing_keys = tuple(
            role for role in RESEARCH_NODE_ROLES if role != NEWS_ANALYSIS
        )
        request = GraphMutationProposalRequest(
            task=state.task.description,
            trigger_node_key=COMPANY_RESEARCH,
            trigger_observation={
                "context_package": package.model_dump(mode="json"),
                "accepted_output_present": output is not None,
            },
            existing_node_keys=existing_keys,
            allowed_new_node_keys=(NEWS_ANALYSIS,),
            available_strategies=(
                RESEARCH_STRATEGY_ID,
                REVIEW_STRATEGY_ID,
            ),
            available_execution_capability_ids=(INFORMATION_RETRIEVAL,),
            evidence=evidence,
        )
        proposal_action_id = uuid4()
        turn = await planner.propose_mutations(
            request,
            invocation=CapabilityInvocationMetadata(
                correlation=InferenceCorrelation(
                    run_id=state.run_id,
                    task_id=state.task.task_id,
                    node_id=node.node_id,
                    action_id=proposal_action_id,
                ),
                trace_attributes={"operation": "graph.mutate.propose"},
            ),
        )
        if turn.kind is CapabilityTurnKind.TOOL_INTENT or turn.result is None:
            raise RuntimeError("Mutation Planner returned no mutation proposal")
        proposal = self._draft_validator.validate_graph_mutation_proposal(
            request,
            turn.result,
        )
        mutations = build_research_mutations_from_draft(
            self._workspace.definition,
            proposal,
        )
        governance_request = GovernanceRequest(
            scope=GovernanceScope.STATE,
            operation="graph.mutate",
            target=GovernanceTarget(
                target_type="task_graph",
                target_id=str(state.run_id),
            ),
            risk=RiskLevel.MEDIUM,
            signals=ConfidenceSignals(
                stated_confidence=0.9,
                evidence=tuple(
                    GovernanceEvidence(
                        evidence_id=item.reference_id,
                        kind=item.kind,
                        source="context_adapter",
                        reliability=item.reliability or 1.0,
                        summary=item.summary or "Accepted Runtime evidence.",
                    )
                    for item in evidence
                ),
                impact=ImpactAssessment(
                    score=0.4,
                    reversible=True,
                    description=(
                        "The proposal changes only this run's validated task graph."
                    ),
                ),
                history=GovernanceHistory(successful_similar=5),
            ),
            correlation=GovernanceCorrelation(
                run_id=state.run_id,
                task_id=state.task.task_id,
                node_id=node.node_id,
                action_id=proposal_action_id,
            ),
            attributes={
                "proposal": proposal.model_dump(mode="json"),
                "planner_capability_id": planner.capability_id,
            },
        )
        record = authorize(governance_request)
        self._workspace.graph_mutation_proposals.append(proposal)
        self._workspace.governance_records.append(record)
        if record.authorization is None:
            raise RuntimeError("Governance denied LLM Graph Mutation Proposal")
        return mutations


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
            goal=node.goal,
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
                "Use only supplied analyses and cite their evidence reference IDs."
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
    selector: ToolSelector,
) -> ToolExecutionStrategy:
    """Small typed seam kept here so Strategies reuse the Runtime adapter."""

    return ToolExecutionStrategy(
        requests=ResearchCapabilityRequestProvider(workspace),
        resolver=resolver,
        selector=selector,
        executor=executor,
        policy=ToolExecutionPolicy(timeout_seconds=2.0),
        strategy_id="research-tool-delegate",
    )
