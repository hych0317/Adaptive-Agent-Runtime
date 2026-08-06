"""Research Agent Application composed exclusively from Runtime public APIs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from collections.abc import Callable
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from adaptive_agent_runtime import (
    AgentRuntime,
    AgentTask,
    InMemoryStateStore,
    InMemoryTraceSink,
    RunResult,
    RunStopPolicy,
    TraceSink,
    RuntimeEvent,
    TraceEntry,
)
from adaptive_agent_runtime.persistence import (
    SQLitePersistence,
    default_runtime_configuration_snapshot,
)
from adaptive_agent_runtime.context_memory import (
    ConditionalMemoryRecall,
    ContextAssembler,
    ContextCompressionExecutionPolicy,
    ContextCompressionDecisionPayload,
    ContextCompressionEffect,
    ContextLayer,
    ContextLifecycleManager,
    ContextLifecycleRuntime,
    ContextMemoryCoordinator,
    ContextMetadata,
    ContextScheduler,
    ContextSource,
    ContextUnit,
    DeterministicContextLifecyclePolicy,
    DeterministicContextPressureMonitor,
    EvidenceDrivenMemoryConsolidator,
    ExperienceAssessmentDraft,
    ExperienceAssessmentRequest,
    ExperienceMetadataEffect,
    InMemoryContextArchive,
    InMemoryContextStore,
    InMemoryMemoryStore,
    MemoryCandidate,
    MemoryCondition,
    MemoryEvidence,
    MemoryEvolutionType,
    MemoryUpdateResult,
    MemoryRecallBundle,
    MemoryRecallDraft,
    MemoryRecallEffect,
    MemoryRecallRequest,
    MemoryScope,
    MemoryExtractionExecutionPolicy,
    MemoryExtractionDecisionPayload,
    MemoryExtractionEffect,
    ResidencyPolicy,
)
from adaptive_agent_runtime.evaluation import (
    AgentEvaluationPipeline,
    ContextMemoryComponentEvaluator,
    ContextMemoryTraceAdapter,
    DeterministicFailureAnalyzer,
    DeterministicOutcomeEvaluator,
    DeterministicTrajectoryEvaluator,
    EvaluationCorrelation,
    EvaluationCriteria,
    EvaluationInputAssembler,
    EvaluationReport,
    GraphSnapshotAdapter,
    InMemoryRootCauseAssessmentStore,
    OrchestrationComponentEvaluator,
    RuntimeNativeTraceCollector,
    RecoveryTraceAdapter,
    RootCauseExecutionPolicy,
    RootCauseDecisionPayload,
    RootCauseAssessmentEffect,
    ToolComponentEvaluator,
)
from adaptive_agent_runtime.governance import (
    BoundGovernedOperation,
    DecisionOutcome,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernanceEvaluator,
    GovernanceHistory,
    GovernanceRequest,
    GovernedOperationExecutor,
    HumanReviewDecision,
    InMemoryHumanReviewService,
    InMemoryAuthorizationConsumptionStore,
    MemoryGovernanceAdapter,
    ReviewOutcome,
    RuntimeGovernanceEvaluator,
    StrictAuthorizationVerifier,
    default_governance_policy,
)
from adaptive_agent_runtime.orchestration import (
    DeterministicFailureDrivenReplanner,
    DynamicTaskGraphPlanner,
    IsolatedAgentExecutor,
    PlanningExecutionPolicy,
    PLANNING_DECISION_TYPE,
    PlanningDecisionPayload,
    PlanningGraphEffect,
    RecoveryExecutionPolicy,
    StrategyActionExecutor,
    TaskGraphStore,
    GraphMutationExecutionPolicy,
    GraphMutationDecisionPayload,
    GraphMutationEffect,
    RecoveryDecisionPayload,
    RecoveryDecisionEffect,
)
from adaptive_agent_runtime.llm import (
    AutonomousAgentBackend,
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    EvidenceReference,
    InferenceCorrelation,
    JudgeAssessmentDraft,
    JudgeRequest,
    TaskGraphDraft,
    CompressedContextDraft,
    RecoveryDraft,
    GraphMutationProposalDraft,
    MemoryCandidateBatchDraft,
    ToolSelectionDraft,
    ActionProposalDraft,
    RootCauseDraft,
)
from adaptive_agent_runtime.tool_ecosystem import (
    ToolExecutionPolicy,
    ToolExecutionStrategy,
    ToolSelectionExecutionPolicy,
    ToolSelectionDecisionPayload,
    ToolSelectionEffect,
    ToolInvocationDecisionPayload,
    ToolInvocationProposalDraft,
    ToolInvocationEffect,
)
from adaptive_agent_runtime.orchestration import (
    ReadyNodeSelectionDecisionPayload,
    ReadyNodeSelectionEffect,
)

from applications.research_agent.capabilities import build_research_tool_stack
from applications.research_agent.cognition import ResearchCognitiveCapabilities
from applications.research_agent.context_compression import (
    DeterministicCompressionProposalCapability,
    ResearchContextCompressionDecisionHandler,
)
from applications.research_agent.graph_mutation import (
    DeterministicResearchGraphMutationCapability,
    ResearchGraphMutationDecisionHandler,
)
from applications.research_agent.memory_extraction import (
    ResearchMemoryExtractionDecisionHandler,
)
from applications.research_agent.memory_recall import (
    DeterministicMemoryRecallCapability,
    ResearchInitialMemoryRecallHandler,
    ResearchMemoryRecallResult,
)
from applications.research_agent.experience_assessment import (
    DeterministicExperienceAssessmentCapability,
    ResearchExperienceAssessmentHandler,
)
from applications.research_agent.decision_feedback import (
    ResearchDecisionFeedbackHandler,
)
from applications.research_agent.experience_learning import (
    DeterministicLearningAssessmentCapability,
    ResearchExperienceLearningHandler,
)
from applications.research_agent.optimization import (
    DeterministicOptimizationAssessmentCapability,
    ResearchOptimizationProposalHandler,
    ResearchPlanningOptimizationBaselineProvider,
    research_initial_planning_optimization_scope,
)
from applications.research_agent.optimization_apply import (
    ResearchOptimizationConfigurationGateway,
    ResearchOptimizationConfigurationResult,
)
from applications.research_agent.auto_adaptation import (
    ResearchAutoAdaptationCoordinator,
    create_unpersisted_auto_adaptation_failure_outcome,
)
from adaptive_agent_runtime.optimization import (
    AutoAdaptationPolicy,
    OptimizationApplyEffect,
    OptimizationApplyIntent,
    OptimizationApplyRequest,
    OptimizationAssessmentRequest,
    OptimizationProposalDraft,
    OptimizationProposalEffect,
    OptimizationRollbackEffect,
    OptimizationRollbackIntent,
    OptimizationRollbackRequest,
    OptimizationTargetKey,
    RuntimeConfigurationSnapshot,
)
from adaptive_agent_runtime.decision_feedback import (
    DecisionFeedbackDraft,
    DecisionFeedbackEffect,
    DecisionFeedbackRequest,
)
from adaptive_agent_runtime.experience_learning import (
    LearningAssessmentRequest,
    LearningInsightDraft,
    LearningInsightEffect,
)
from applications.research_agent.ready_node_selection import (
    LLMReadyTaskNodeSelector,
)
from applications.research_agent.report import (
    GovernanceRecord,
    ResearchReport,
    ResearchRunResult,
)
from applications.research_agent.report_decision import (
    ReportArtifactEffect,
    ReportDecisionPayload,
    ReportDraft,
    ResearchReportDecisionHandler,
)
from adaptive_agent_runtime.decisioning import DecisionCheckpoint, decision_fingerprint
from applications.research_agent.progress import (
    ObservableTraceSink,
    ResearchProgressEvent,
    ResearchProgressKind,
    ResearchProgressSink,
    publish_progress,
)
from applications.research_agent.planning import (
    DeterministicResearchPlanningCapability,
    run_adaptive_planning,
)
from applications.research_agent.persistence import (
    ResearchPersistenceIdentity,
    SQLiteReportDispatchReconciler,
    SQLiteResearchRunManifestStore,
)
from applications.research_agent.recovery import ResearchRecoveryDecisionHandler
from applications.research_agent.root_cause import (
    ResearchRootCauseDecisionHandler,
    ResearchRootCauseRecoveryBridge,
)
from applications.research_agent.tool_selection import (
    ResearchToolSelectionDecisionHandler,
)
from applications.research_agent.tool_invocation import (
    ResearchToolInvocationDecisionHandler,
)
from applications.research_agent.prompts import CHINESE_OUTPUT_INSTRUCTION
from applications.research_agent.strategies import (
    DeterministicRiskAgent,
    GovernedRecordingToolExecutor,
    GovernedContextLifecycleExecutor,
    ReportStrategy,
    ResearchStrategy,
    ResearchContextRequirementProvider,
    ResearchRecoveryPlanApplier,
    ResearchWorkspace,
    ReviewStrategy,
    LLMRiskAgent,
    build_tool_execution_strategy,
)
from applications.research_agent.tasks import (
    REPORT_GENERATION,
    REPORT_STRATEGY_ID,
    RESEARCH_STRATEGY_ID,
    REVIEW_STRATEGY_ID,
    ResearchTaskDefinition,
)


def company_from_task(task: str) -> str:
    """Extract a useful company label from the documented demo input form."""

    value = task.strip()
    for prefix in ("请分析", "分析"):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
            break
    for suffix in ("的投资价值", "投资价值"):
        if value.endswith(suffix):
            value = value[: -len(suffix)].strip()
            break
    return value or task.strip() or "Unknown Company"


class ResearchInformationMode(StrEnum):
    """Application-owned choice of evidence Provider composition."""

    FIXTURE_DEMO = "fixture_demo"
    LLM_RESEARCH = "llm_research"


class _PersistentTraceMirror:
    """Keep synchronous run-local reads while every event is durably recorded."""

    module_id = "research_agent.trace.persistent_mirror"

    def __init__(self, local: InMemoryTraceSink, durable: TraceSink) -> None:
        self._local = local
        self._durable = durable

    async def record(self, event: RuntimeEvent) -> TraceEntry:
        durable = await self._durable.record(event)
        await self._local.record(event)
        return durable


DEFAULT_RESEARCH_RUNTIME_DB = (
    Path(__file__).resolve().parents[2] / "data" / "research_runtime.sqlite3"
)


class ResearchAgent:
    """Application facade demonstrating the full Adaptive Runtime stack."""

    @classmethod
    def for_runtime(cls, **kwargs: Any) -> ResearchAgent:
        """Create a real Runtime instance backed by the production database."""

        return cls(
            persistence_path=DEFAULT_RESEARCH_RUNTIME_DB,
            run_kind="runtime",
            disposable=False,
            **kwargs,
        )

    def __init__(
        self,
        *,
        persistence_path: str | Path,
        run_kind: str = "runtime",
        disposable: bool = False,
        autonomous_risk_backend: AutonomousAgentBackend | None = None,
        cognitive_capabilities: ResearchCognitiveCapabilities | None = None,
        planning_execution_policy: PlanningExecutionPolicy | None = None,
        information_mode: ResearchInformationMode = (
            ResearchInformationMode.FIXTURE_DEMO
        ),
        decision_fault_injector: Callable[..., None] | None = None,
        auto_adaptation_enabled: bool = False,
        run_stop_policy: RunStopPolicy | None = None,
    ) -> None:
        if run_kind not in {"runtime", "test", "demo", "benchmark"}:
            raise ValueError("unsupported Research run kind")
        if run_kind == "runtime" and disposable:
            raise ValueError("real Runtime runs cannot be disposable")
        self._run_kind = run_kind
        self._disposable = disposable
        self._run_stop_policy = run_stop_policy or RunStopPolicy(
            max_action_steps=20
        )
        self._cognitive_capabilities = (
            cognitive_capabilities or ResearchCognitiveCapabilities()
        )
        if (
            information_mode is ResearchInformationMode.LLM_RESEARCH
            and self._cognitive_capabilities.report_generator is None
        ):
            raise ValueError(
                "LLM Research mode requires the generation capability"
            )
        self._information_mode = information_mode
        self._decision_fault_injector = decision_fault_injector
        self._auto_adaptation_policy = AutoAdaptationPolicy(
            enabled=auto_adaptation_enabled,
            scope=research_initial_planning_optimization_scope(),
        )
        planner_capability = self._cognitive_capabilities.task_planner
        self._planning_execution_policy = (
            planning_execution_policy
            or PlanningExecutionPolicy(
                model=(
                    f"{planner_capability.capability_id}:configured"
                    if planner_capability is not None
                    else "runtime.static"
                )
            )
        )
        self._persistence = SQLitePersistence(persistence_path)
        self._tools = build_research_tool_stack(
            llm_information_generator=(
                self._cognitive_capabilities.report_generator
                if information_mode is ResearchInformationMode.LLM_RESEARCH
                else None
            ),
            permit_verifier=self._persistence.commit_permit_verifier,
        )
        self._manifest_store = SQLiteResearchRunManifestStore(
            self._persistence.database
        )
        self._report_dispatch_reconciler = SQLiteReportDispatchReconciler(
            self._persistence.database,
            self._persistence.decision_records,
        )
        self._context_store = self._persistence.context_store
        self._context_archive = self._persistence.context_archive
        self._memory_store = self._persistence.memory_store
        self._memory_consolidator = EvidenceDrivenMemoryConsolidator(
            self._memory_store
        )
        self._reviews = self._persistence.human_review_service
        self._governance: GovernanceEvaluator = RuntimeGovernanceEvaluator(
            policy=default_governance_policy(),
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=self._reviews,
        )
        self._issuer = self._persistence.authorization_issuer
        self._authorization_store = self._persistence.authorization_store
        self._operation_executor = self._persistence.operation_executor
        self._autonomous_risk_backend = autonomous_risk_backend
        self._optimization_configuration_gateway = (
            self._build_optimization_configuration_gateway(
                self._persistence.trace_sink
            )
        )

    async def run(
        self,
        task: str,
        *,
        progress_sink: ResearchProgressSink | None = None,
        _resume_run_id: UUID | None = None,
    ) -> ResearchRunResult:
        """Execute one research run and evaluate it after Runtime completion."""

        resuming = _resume_run_id is not None
        memory_baseline = {
            item.memory_id: decision_fingerprint(item)
            for item in await self._memory_store.list_all()
        }
        graph_store: TaskGraphStore | None
        if resuming:
            assert _resume_run_id is not None
            restored_state = await self._persistence.state_store.load(_resume_run_id)
            manifest = self._manifest_store.load(_resume_run_id)
            if restored_state is None or manifest is None:
                raise RuntimeError("Research run cannot resume without State and manifest")
            task, restored_definition, restored_configuration = manifest
            agent_task = restored_state.task
            company = restored_definition.company
            run_id = _resume_run_id
            configuration_snapshot = (
                restored_configuration
                or default_runtime_configuration_snapshot(
                    research_initial_planning_optimization_scope()
                )
            )
        else:
            company = company_from_task(task)
            run_id = uuid4()
            agent_task = AgentTask(
                description=task,
                input={"company": company, "application": "research_agent"},
            )
            configuration_snapshot = (
                await self._persistence.runtime_configuration.load_active(
                    research_initial_planning_optimization_scope(),
                    OptimizationTargetKey.PLANNER_MAX_NODES,
                )
                or default_runtime_configuration_snapshot(
                    research_initial_planning_optimization_scope()
                )
            )
        await publish_progress(
            progress_sink,
            ResearchProgressEvent(
                run_id=run_id,
                kind=ResearchProgressKind.RUN_PREPARING,
                payload={"task": task, "company": company},
            ),
        )
        runtime_trace_sink = InMemoryTraceSink()
        if resuming:
            for entry in await self._persistence.trace_sink.entries_for(run_id):
                await runtime_trace_sink.record(entry.event)
        runtime_trace_writer: TraceSink = _PersistentTraceMirror(
            runtime_trace_sink,
            self._persistence.trace_sink,
        )
        if progress_sink is not None:
            runtime_trace_writer = ObservableTraceSink(
                runtime_trace_writer,
                progress_sink,
            )
        root_cause_store = InMemoryRootCauseAssessmentStore()
        root_cause_handler = (
            ResearchRootCauseDecisionHandler(
                capability=self._cognitive_capabilities.root_cause_analyzer,
                execution_policy=RootCauseExecutionPolicy(),
                governance=self._governance,
                reviews=self._reviews,
                issuer=self._issuer,
                operation_executor=self._operation_executor,
                trace_sink=runtime_trace_writer,
                assessment_store=root_cause_store,
                checkpoint_store=self._persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        RootCauseDecisionPayload,
                        RootCauseDraft,
                        RootCauseAssessmentEffect,
                    ]
                ),
            )
            if self._cognitive_capabilities.root_cause_analyzer is not None
            else None
        )
        root_cause_recovery_bridge = (
            ResearchRootCauseRecoveryBridge(
                handler=root_cause_handler,
                assessment_store=root_cause_store,
                trace_reader=runtime_trace_sink.entries_for,
            )
            if root_cause_handler is not None
            else None
        )
        if resuming:
            definition = restored_definition
            llm_task_graph_draft = None
            planning_record = None
            recall_result = None
            recall_bundle = await self._persistence.memory_recall_bundle_store.load_for_run(
                run_id
            )
            graph_store = self._persistence.task_graph_store
            await self._report_dispatch_reconciler.reconcile(run_id)
        else:
            (
                definition,
                llm_task_graph_draft,
                planning_record,
                graph_store,
                recall_result,
            ) = (
                await self._build_task_definition(
                    company=company,
                    task=task,
                    run_id=run_id,
                    agent_task=agent_task,
                    trace_sink=runtime_trace_writer,
                    configuration_snapshot=configuration_snapshot,
                )
            )
            recall_bundle = (
                recall_result.bundle if recall_result is not None else None
            )
            self._manifest_store.save(
                run_id=run_id,
                task=task,
                definition=definition,
                configuration_snapshot=configuration_snapshot,
                run_kind=self._run_kind,
                disposable=self._disposable,
            )
        await publish_progress(
            progress_sink,
            ResearchProgressEvent(
                run_id=run_id,
                kind=ResearchProgressKind.GRAPH_INITIALIZED,
                payload={
                    "graph": definition.initial_graph.model_dump(mode="json")
                },
            ),
        )
        workspace = ResearchWorkspace(definition)
        if resuming:
            graph_checkpoint = await self._persistence.task_graph_store.load(run_id)
            if graph_checkpoint is None:
                raise RuntimeError("Research run has no durable Graph checkpoint")
            for node in graph_checkpoint.graph.nodes:
                if node.observation is not None and node.observation.succeeded:
                    if workspace.role_for(node) == REPORT_GENERATION:
                        persisted_report = self._persistence.workspace_artifact_store.load(
                            run_id=run_id,
                            node_id=node.node_id,
                            artifact_type="research_report",
                        )
                        if persisted_report is None:
                            raise RuntimeError(
                                "completed report node has no authoritative artifact receipt"
                            )
                        workspace.restore_report_commit(
                            node, persisted_report[0], persisted_report[1]
                        )
                    else:
                        workspace.set_output(node, node.observation.output)
        configured_compressor = self._cognitive_capabilities.context_compressor
        context_lifecycle = ContextLifecycleManager(
            store=self._context_store,
            archive=self._context_archive,
            compressor=ResearchContextCompressionDecisionHandler(
                capability=(
                    configured_compressor
                    or DeterministicCompressionProposalCapability()
                ),
                store=self._context_store,
                archive=self._context_archive,
                execution_policy=ContextCompressionExecutionPolicy(),
                governance=self._governance,
                reviews=self._reviews,
                issuer=self._issuer,
                operation_executor=self._operation_executor,
                trace_sink=runtime_trace_writer,
                workspace=workspace,
                context_projection=(
                    self._cognitive_capabilities.compression_context
                    if configured_compressor is not None
                    else None
                ),
                checkpoint_store=self._persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        ContextCompressionDecisionPayload,
                        CompressedContextDraft,
                        ContextCompressionEffect,
                    ]
                ),
            ),
        )
        context_coordinator = ContextMemoryCoordinator(
            context_store=self._context_store,
            scheduler=ContextScheduler(),
            assembler=ContextAssembler(),
            requirements=ResearchContextRequirementProvider(
                workspace,
                max_units=64,
                max_tokens=8192,
            ),
            # Phase 3-A permits persisted Recall only once, before Planning.
            # Node execution Context must not perform an ungoverned dynamic recall.
            memory_recall=ConditionalMemoryRecall(InMemoryMemoryStore()),
            lifecycle=ContextLifecycleRuntime(
                store=self._context_store,
                pressure_monitor=DeterministicContextPressureMonitor(
                    max_resident_tokens=768,
                ),
                policy=DeterministicContextLifecyclePolicy(
                    max_compressed_units=1,
                ),
                executor=GovernedContextLifecycleExecutor(
                    manager=context_lifecycle,
                    authorize=lambda request: self._authorize_with_demo_review(
                        request,
                        scenario="context_lifecycle",
                    ),
                    operation_executor=self._operation_executor,
                    workspace=workspace,
                ),
                max_passes=2,
            ),
        )
        if planning_record is not None:
            workspace.governance_records.append(planning_record)
        if recall_result is not None and recall_result.governance_record is not None:
            workspace.governance_records.append(recall_result.governance_record)
        context_trace = ContextMemoryTraceAdapter()
        goal_context = ContextUnit(
            content={"research_goal": task, "company": company},
            metadata=ContextMetadata(
                source=ContextSource.CONVERSATION,
                layer=ContextLayer.WORKING,
                run_id=run_id,
                task_id=agent_task.task_id,
                source_reference=f"task:{agent_task.task_id}",
                tags=("research", "goal", "financial_research"),
                importance=1.0,
                estimated_tokens=24,
            ),
            residency_policy=ResidencyPolicy.PINNED,
        )
        if resuming:
            workspace.context_units.extend(
                await self._context_store.list_for_run(run_id)
            )
        else:
            await context_lifecycle.add(goal_context)
            workspace.context_units.append(goal_context)
            workspace.trace_batches.append(
                context_trace.context_unit(goal_context, kind="context.research_goal")
            )
            if not await self._memory_store.list_all():
                await self._seed_memories(goal_context, workspace, context_trace)
            else:
                await self._reinforce_memory(goal_context, workspace, context_trace)

        governed_executor = GovernedRecordingToolExecutor(
            delegate=self._tools.executor,
            governance=self._governance,
            reviews=self._reviews,
            issuer=self._issuer,
            operation_executor=self._operation_executor,
            workspace=workspace,
        )
        tool_selection_handler = (
            ResearchToolSelectionDecisionHandler(
                capability=self._cognitive_capabilities.tool_selector,
                resolver=self._tools.resolver,
                execution_policy=ToolSelectionExecutionPolicy(),
                governance=self._governance,
                reviews=self._reviews,
                issuer=self._issuer,
                operation_executor=self._operation_executor,
                trace_sink=runtime_trace_writer,
                workspace=workspace,
                checkpoint_store=self._persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        ToolSelectionDecisionPayload,
                        ToolSelectionDraft,
                        ToolSelectionEffect,
                    ]
                ),
            )
            if self._cognitive_capabilities.tool_selector is not None
            else None
        )
        tool_strategy: ToolExecutionStrategy = build_tool_execution_strategy(
            workspace=workspace,
            executor=governed_executor,
            resolver=self._tools.resolver,
            selector=(
                self._tools.selector if tool_selection_handler is None else None
            ),
            selection_handler=tool_selection_handler,
            timeout_seconds=(
                300.0
                if self._information_mode
                is ResearchInformationMode.LLM_RESEARCH
                else 2.0
            ),
        )
        research_strategy = ResearchStrategy(
            tool_strategy=tool_strategy,
            workspace=workspace,
            lifecycle=context_lifecycle,
            coordinator=context_coordinator,
            context_trace=context_trace,
            reasoner=self._cognitive_capabilities.reasoner,
            mutation_planner=self._cognitive_capabilities.mutation_planner,
            tool_invocation_handler=(
                ResearchToolInvocationDecisionHandler(
                    resolver=self._tools.resolver,
                    selector=(
                        self._tools.selector
                        if tool_selection_handler is None
                        else None
                    ),
                    selection_handler=tool_selection_handler,
                    executor=self._tools.executor,
                    governance=self._governance,
                    reviews=self._reviews,
                    issuer=self._issuer,
                    operation_executor=self._operation_executor,
                    trace_sink=runtime_trace_writer,
                    workspace=workspace,
                    checkpoint_store=self._persistence.create_decision_checkpoint_store(
                        DecisionCheckpoint[
                            ToolInvocationDecisionPayload,
                            ToolInvocationProposalDraft,
                            ToolInvocationEffect,
                        ]
                    ),
                    execution_policy=ToolExecutionPolicy(
                        timeout_seconds=(
                            300.0
                            if self._information_mode
                            is ResearchInformationMode.LLM_RESEARCH
                            else 2.0
                        )
                    ),
                )
                if self._cognitive_capabilities.reasoner_tool_intent_limit
                else None
            ),
            max_reasoning_tool_intents=(
                self._cognitive_capabilities.reasoner_tool_intent_limit
            ),
        )
        report_strategy = ReportStrategy(
            tool_strategy=tool_strategy,
            workspace=workspace,
            lifecycle=context_lifecycle,
            coordinator=context_coordinator,
            context_trace=context_trace,
            llm_generator=self._cognitive_capabilities.report_generator,
            llm_context=self._cognitive_capabilities.report_context,
            decision_handler=ResearchReportDecisionHandler(
                committer=self._persistence.workspace_artifact_committer,
                governance=self._governance,
                reviews=self._reviews,
                issuer=self._issuer,
                operation_executor=self._operation_executor,
                trace_sink=runtime_trace_writer,
                checkpoint_store=self._persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        ReportDecisionPayload,
                        ReportDraft,
                        ReportArtifactEffect,
                    ]
                ),
                fault_injector=self._decision_fault_injector,
            ),
        )
        isolated_directory: TemporaryDirectory[str] | None = None
        try:
            isolated_executor: IsolatedAgentExecutor
            if self._autonomous_risk_backend is None:
                isolated_executor = DeterministicRiskAgent(workspace)
            else:
                isolated_directory = TemporaryDirectory(
                    prefix="aar-research-agent-"
                )
                isolated_executor = LLMRiskAgent(
                    backend=self._autonomous_risk_backend,
                    workspace=workspace,
                    isolated_workspace_root=isolated_directory.name,
                    authorize=lambda request: self._authorize_with_demo_review(
                        request,
                        scenario="agent",
                    ),
                    operation_executor=self._operation_executor,
                )
            review_strategy = ReviewStrategy(
                isolated_executor=isolated_executor,
                workspace=workspace,
                lifecycle=context_lifecycle,
                coordinator=context_coordinator,
                context_trace=context_trace,
            )
            action_planner = self._cognitive_capabilities.action_planner
            ready_node_selector = (
                LLMReadyTaskNodeSelector(
                    capability=action_planner,
                    workspace=workspace,
                    governance=self._governance,
                    reviews=self._reviews,
                    issuer=self._issuer,
                    operation_executor=self._operation_executor,
                    trace_sink=runtime_trace_writer,
                    checkpoint_store=self._persistence.create_decision_checkpoint_store(
                        DecisionCheckpoint[
                            ReadyNodeSelectionDecisionPayload,
                            ActionProposalDraft,
                            ReadyNodeSelectionEffect,
                        ]
                    ),
                )
                if action_planner is not None
                else None
            )
            mutation_capability = self._cognitive_capabilities.mutation_planner
            mutation_decision_handler = ResearchGraphMutationDecisionHandler(
                capability=(
                    mutation_capability
                    if mutation_capability is not None
                    else DeterministicResearchGraphMutationCapability(workspace)
                ),
                context_projection=self._cognitive_capabilities.mutation_context,
                execution_policy=GraphMutationExecutionPolicy(),
                governance=self._governance,
                reviews=self._reviews,
                issuer=self._issuer,
                operation_executor=self._operation_executor,
                trace_sink=runtime_trace_writer,
                workspace=workspace,
                record_agent_proposal=mutation_capability is not None,
                checkpoint_store=self._persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        GraphMutationDecisionPayload,
                        GraphMutationProposalDraft,
                        GraphMutationEffect,
                    ]
                ),
            )
            planner = DynamicTaskGraphPlanner(
                definition.initial_graph,
                graph_store=graph_store,
                commit_permit_verifier=self._persistence.commit_permit_verifier,
                ready_node_selector=ready_node_selector,
                mutation_applier=None,
                mutation_decision_handler=mutation_decision_handler,
                recovery_planner=(
                    None
                    if self._cognitive_capabilities.recovery_planner is not None
                    else DeterministicFailureDrivenReplanner(
                        max_attempts_per_node=2,
                    )
                ),
                recovery_applier=(
                    None
                    if self._cognitive_capabilities.recovery_planner is not None
                    else ResearchRecoveryPlanApplier(
                        authorize=lambda request: self._authorize_with_demo_review(
                            request,
                            scenario="failure_recovery",
                        ),
                        operation_executor=self._operation_executor,
                        workspace=workspace,
                    )
                ),
                recovery_decision_handler=(
                    ResearchRecoveryDecisionHandler(
                        capability=self._cognitive_capabilities.recovery_planner,
                        execution_policy=RecoveryExecutionPolicy(
                            max_recovery_attempts=2,
                        ),
                        governance=self._governance,
                        reviews=self._reviews,
                        issuer=self._issuer,
                        operation_executor=self._operation_executor,
                        trace_sink=runtime_trace_writer,
                        workspace=workspace,
                        available_strategy_ids=(
                            RESEARCH_STRATEGY_ID,
                            REVIEW_STRATEGY_ID,
                            REPORT_STRATEGY_ID,
                        ),
                        diagnostic_evidence_provider=(
                            root_cause_recovery_bridge
                        ),
                        checkpoint_store=self._persistence.create_decision_checkpoint_store(
                            DecisionCheckpoint[
                                RecoveryDecisionPayload,
                                RecoveryDraft,
                                RecoveryDecisionEffect,
                            ]
                        ),
                    )
                    if self._cognitive_capabilities.recovery_planner is not None
                    else None
                ),
            )
            runtime = AgentRuntime(
                planner=planner,
                executor=StrategyActionExecutor(
                    (research_strategy, review_strategy, report_strategy)
                ),
                state_store=self._persistence.state_store,
                trace_sink=runtime_trace_writer,
                stop_policy=self._run_stop_policy,
            )
            runtime_result = (
                await runtime.resume(run_id)
                if resuming
                else await runtime.run(agent_task, run_id=run_id)
            )
        finally:
            if isolated_directory is not None:
                isolated_directory.cleanup()
        final_state = runtime_result.final_state
        resumed_graph_checkpoint = (
            await self._persistence.task_graph_store.load(run_id)
            if resuming
            else None
        )
        final_graph = (
            resumed_graph_checkpoint.graph
            if resumed_graph_checkpoint is not None
            else planner.graph_for(run_id)
        )
        recovery_records = (
            resumed_graph_checkpoint.recovery_records
            if resumed_graph_checkpoint is not None
            else planner.recovery_records_for(run_id)
        )

        memory_extractor = self._cognitive_capabilities.memory_extractor
        if memory_extractor is not None and not resuming:
            extraction_context = self._cognitive_capabilities.extraction_context
            if extraction_context is None:
                raise RuntimeError(
                    "Memory Extraction Decision requires Context projection"
                )
            await ResearchMemoryExtractionDecisionHandler(
                capability=memory_extractor,
                context_projection=extraction_context,
                memory_store=self._memory_store,
                consolidator=self._memory_consolidator,
                execution_policy=MemoryExtractionExecutionPolicy(),
                governance=self._governance,
                reviews=self._reviews,
                issuer=self._issuer,
                operation_executor=self._operation_executor,
                trace_sink=runtime_trace_writer,
                workspace=workspace,
                checkpoint_store=self._persistence.create_decision_checkpoint_store(
                    DecisionCheckpoint[
                        MemoryExtractionDecisionPayload,
                        MemoryCandidateBatchDraft,
                        MemoryExtractionEffect,
                    ]
                ),
            ).extract(
                run_id=run_id,
                task_id=agent_task.task_id,
                state_revision=final_state.revision,
                context_trace=context_trace,
            )
        runtime_entries = runtime_trace_sink.entries_for(run_id)
        tool_entries = tuple(
            entry
            for observation in workspace.tool_observations
            for entry in self._tools.trace_sink.entries_for(
                observation.invocation_id
            )
        )
        extra_batches = [
            *workspace.trace_batches,
            GraphSnapshotAdapter().adapt(
                final_graph,
                run_id=run_id,
                task_id=agent_task.task_id,
                observed_at=final_state.updated_at,
            ),
            RecoveryTraceAdapter().adapt(
                recovery_records,
                run_id=run_id,
                task_id=agent_task.task_id,
                graph_id=final_graph.graph_id,
            ),
        ]
        subject = EvaluationInputAssembler(
            collector=RuntimeNativeTraceCollector()
        ).assemble(
            runtime_result,
            runtime_entries=runtime_entries,
            tool_entries=tool_entries,
            extra_batches=extra_batches,
        )
        evaluation = AgentEvaluationPipeline(
            outcome=DeterministicOutcomeEvaluator(),
            trajectory=DeterministicTrajectoryEvaluator(),
            components=(
                OrchestrationComponentEvaluator(),
                ToolComponentEvaluator(),
                ContextMemoryComponentEvaluator(),
            ),
        ).evaluate(
            subject,
            EvaluationCriteria(required_output_keys=("graph_id", "nodes")),
        )
        if resuming:
            persisted_evaluation = (
                await self._persistence.evaluation_report_store.load_for_run(run_id)
            )
            evaluation = (
                persisted_evaluation
                if persisted_evaluation is not None
                else await self._persistence.evaluation_report_store.save(evaluation)
            )
        else:
            evaluation = await self._persistence.evaluation_report_store.save(
                evaluation
            )
        if root_cause_handler is not None and not resuming:
            await root_cause_handler.analyze_post_run(
                evaluation,
                subject.trace,
            )
        llm_judgement = (
            None
            if resuming
            else await self._judge_evaluation(
                evaluation,
                workspace,
                runtime_result,
            )
        )
        # Deterministic failure analysis remains read-only Evaluation output.  It
        # is no longer accumulated in process and cannot create a Proposal.
        failure_analysis = DeterministicFailureAnalyzer().analyze(
            evaluation.results
        )

        raw_report = workspace.output_for_role(REPORT_GENERATION)
        if raw_report is None:
            raise RuntimeError(
                "Research Runtime produced no report: "
                + (final_state.error or final_state.status.value)
            )
        report = ResearchReport.model_validate(raw_report)
        report_node_id = workspace.definition.node(REPORT_GENERATION).node_id
        report_commit_receipt = workspace.report_commit_receipts.get(report_node_id)
        if report_commit_receipt is None:
            raise RuntimeError("Research report has no authoritative commit receipt")
        memories = await self._memory_store.list_all()
        run_evidence_refs = {
            f"tool:{observation.invocation_id}"
            for observation in workspace.tool_observations
        }
        experience_source_memories = tuple(
            memory
            for memory in memories
            if (
                memory_baseline.get(memory.memory_id)
                != decision_fingerprint(memory)
                or any(
                    evidence.source_reference in run_evidence_refs
                    for evidence in memory.evidence
                )
            )
        )
        experience_handler = ResearchExperienceAssessmentHandler(
            state_store=self._persistence.state_store,
            artifact_readback=self._persistence.workspace_artifact_committer,
            metadata_store=self._persistence.experience_metadata_store,
            capability=(
                self._cognitive_capabilities.experience_assessor
                or DeterministicExperienceAssessmentCapability()
            ),
            governance=self._governance,
            reviews=self._reviews,
            issuer=self._issuer,
            operation_executor=self._operation_executor,
            trace_sink=runtime_trace_writer,
            checkpoint_store=self._persistence.create_decision_checkpoint_store(
                DecisionCheckpoint[
                    ExperienceAssessmentRequest,
                    ExperienceAssessmentDraft,
                    ExperienceMetadataEffect,
                ]
            ),
            fault_injector=self._decision_fault_injector,
        )
        experience_result = (
            await experience_handler.resume(run_id) if resuming else None
        )
        if experience_result is None:
            existing_experience = (
                await self._persistence.experience_metadata_store.list_for_run(run_id)
            )
            if existing_experience:
                experience_metadata = existing_experience[-1]
            else:
                experience_result = await experience_handler.assess(
                    final_state=final_state,
                    evaluation=evaluation,
                    artifact_receipt=report_commit_receipt,
                    source_memories=experience_source_memories,
                )
                experience_metadata = experience_result.metadata
        else:
            experience_metadata = experience_result.metadata
        if (
            experience_result is not None
            and experience_result.governance_record is not None
        ):
            workspace.governance_records.append(
                experience_result.governance_record
            )
        feedback_handler = ResearchDecisionFeedbackHandler(
            state_store=self._persistence.state_store,
            evaluation_store=self._persistence.evaluation_report_store,
            feedback_store=self._persistence.decision_feedback_store,
            governance=self._governance,
            reviews=self._reviews,
            issuer=self._issuer,
            operation_executor=self._operation_executor,
            trace_sink=runtime_trace_writer,
            checkpoint_store=self._persistence.create_decision_checkpoint_store(
                DecisionCheckpoint[
                    DecisionFeedbackRequest,
                    DecisionFeedbackDraft,
                    DecisionFeedbackEffect,
                ]
            ),
            fault_injector=self._decision_fault_injector,
        )
        feedback_result = await feedback_handler.record_for_run(
            final_state=final_state,
            evaluation=evaluation,
            artifact_receipt=report_commit_receipt,
            experience=experience_metadata,
            recall_bundle=recall_bundle,
        )
        workspace.governance_records.extend(feedback_result.governance_records)
        learning_handler = ResearchExperienceLearningHandler(
            store=self._persistence.learning_insight_store,
            capability=(
                self._cognitive_capabilities.experience_learner
                or DeterministicLearningAssessmentCapability()
            ),
            governance=self._governance,
            reviews=self._reviews,
            issuer=self._issuer,
            operation_executor=self._operation_executor,
            trace_sink=runtime_trace_writer,
            checkpoint_store=self._persistence.create_decision_checkpoint_store(
                DecisionCheckpoint[
                    LearningAssessmentRequest,
                    LearningInsightDraft,
                    LearningInsightEffect,
                ]
            ),
            fault_injector=self._decision_fault_injector,
        )
        learning_result = await learning_handler.assess(
            trigger_run_id=run_id,
            task_id=final_state.task.task_id,
            scope=MemoryScope(
                tenant_id="default",
                project_id="research",
                agent_scope="planner",
            ),
            subject_decision_type=PLANNING_DECISION_TYPE,
        )
        if learning_result.governance_record is not None:
            workspace.governance_records.append(
                learning_result.governance_record
            )
        optimization_handler = ResearchOptimizationProposalHandler(
            evidence_resolver=self._persistence.optimization_evidence_resolver,
            baseline_provider=ResearchPlanningOptimizationBaselineProvider(
                self._persistence.runtime_configuration
            ),
            store=self._persistence.optimization_proposal_store,
            capability=(
                self._cognitive_capabilities.optimization_assessor
                or DeterministicOptimizationAssessmentCapability()
            ),
            governance=self._governance,
            reviews=self._reviews,
            issuer=self._issuer,
            operation_executor=self._operation_executor,
            trace_sink=runtime_trace_writer,
            checkpoint_store=self._persistence.create_decision_checkpoint_store(
                DecisionCheckpoint[
                    OptimizationAssessmentRequest,
                    OptimizationProposalDraft,
                    OptimizationProposalEffect,
                ]
            ),
            fault_injector=self._decision_fault_injector,
        )
        optimization_result = await optimization_handler.assess(
            trigger_run_id=run_id,
            task_id=final_state.task.task_id,
            scope=research_initial_planning_optimization_scope(),
        )
        if optimization_result.governance_record is not None:
            workspace.governance_records.append(
                optimization_result.governance_record
            )
        proposals = (
            (optimization_result.proposal,)
            if optimization_result.proposal is not None
            else ()
        )
        # Everything above this point belongs to the completed Research Run.
        # Freeze those outputs before entering the optional post-run control plane.
        completed_governance_records = tuple(workspace.governance_records)
        authorization_uses_list = []
        for record in completed_governance_records:
            if record.authorization is None:
                continue
            use = await self._authorization_store.load(
                record.authorization.authorization_id
            )
            if use is not None:
                authorization_uses_list.append(use)
        completed_authorization_uses = tuple(authorization_uses_list)
        root_cause_assessments = await root_cause_store.list_for_run(run_id)
        completed_runtime_entries = runtime_trace_sink.entries_for(run_id)

        auto_adaptation_coordinator: ResearchAutoAdaptationCoordinator | None = None
        try:
            auto_adaptation_coordinator = ResearchAutoAdaptationCoordinator(
                policy=self._auto_adaptation_policy,
                proposals=self._persistence.optimization_proposal_store,
                configurations=self._persistence.runtime_configuration,
                state_store=self._persistence.state_store,
                triggers=self._persistence.auto_adaptation_trigger_store,
                gateway=self._build_optimization_configuration_gateway(
                    runtime_trace_writer
                ),
                trace_sink=runtime_trace_writer,
            )
            auto_adaptation = (
                await auto_adaptation_coordinator.evaluate_after_run(
                    trigger_run_id=run_id,
                    run_configuration=configuration_snapshot,
                )
            )
        except Exception as error:
            if auto_adaptation_coordinator is None:
                auto_adaptation = (
                    create_unpersisted_auto_adaptation_failure_outcome(
                        policy=self._auto_adaptation_policy,
                        trigger_run_id=run_id,
                        run_configuration=configuration_snapshot,
                        error=error,
                    )
                )
            else:
                try:
                    auto_adaptation = (
                        await auto_adaptation_coordinator
                        .contain_infrastructure_failure(
                            trigger_run_id=run_id,
                            run_configuration=configuration_snapshot,
                            error=error,
                        )
                    )
                except Exception as containment_error:
                    auto_adaptation = (
                        create_unpersisted_auto_adaptation_failure_outcome(
                            policy=self._auto_adaptation_policy,
                            trigger_run_id=run_id,
                            run_configuration=configuration_snapshot,
                            error=error,
                        ).model_copy(
                            update={
                                "error_summary": (
                                    f"{type(error).__name__}: {error}; "
                                    f"containment failed: "
                                    f"{type(containment_error).__name__}: "
                                    f"{containment_error}"
                                )[:1024]
                            }
                        )
                    )

        final_governance_records = completed_governance_records
        final_authorization_uses = completed_authorization_uses
        if (
            auto_adaptation_coordinator is not None
            and auto_adaptation_coordinator.governance_record is not None
        ):
            auto_governance = auto_adaptation_coordinator.governance_record
            final_governance_records = (
                *completed_governance_records,
                auto_governance,
            )
            if auto_governance.authorization is not None:
                try:
                    auto_use = await self._authorization_store.load(
                        auto_governance.authorization.authorization_id
                    )
                except Exception as error:
                    diagnostic = (
                        f"authorization readback failed: "
                        f"{type(error).__name__}: {error}"
                    )[:1024]
                    auto_adaptation = auto_adaptation.model_copy(
                        update={
                            "error_summary": (
                                f"{auto_adaptation.error_summary}; {diagnostic}"
                                if auto_adaptation.error_summary
                                else diagnostic
                            )[:1024]
                        }
                    )
                else:
                    if auto_use is not None:
                        final_authorization_uses = (
                            *completed_authorization_uses,
                            auto_use,
                        )
        runtime_entries = runtime_trace_sink.entries_for(run_id)
        if not runtime_entries:
            runtime_entries = completed_runtime_entries
        return ResearchRunResult(
            runtime_result=runtime_result,
            task_graph=final_graph,
            report=report,
            report_commit_receipt=report_commit_receipt,
            evaluation=evaluation,
            failure_analysis=failure_analysis,
            optimization_proposals=proposals,
            runtime_configuration=configuration_snapshot,
            auto_adaptation=auto_adaptation,
            governance_records=final_governance_records,
            authorization_uses=final_authorization_uses,
            runtime_trace=runtime_entries,
            tool_trace=tool_entries,
            tool_observations=tuple(workspace.tool_observations),
            context_units=tuple(workspace.context_units),
            context_assemblies=tuple(workspace.context_assemblies),
            memories=memories,
            memory_recall_bundle=recall_bundle,
            experience_metadata=experience_metadata,
            decision_feedback=feedback_result.records,
            learning_insights=(
                (learning_result.insight,)
                if learning_result.insight is not None
                else ()
            ),
            agent_executions=tuple(workspace.agent_executions),
            llm_judgement=llm_judgement,
            llm_task_graph_draft=llm_task_graph_draft,
            llm_action_proposals=tuple(workspace.action_proposals),
            llm_graph_mutation_proposals=tuple(
                workspace.graph_mutation_proposals
            ),
            llm_reasoning=tuple(workspace.reasoning_records),
            llm_memory_candidates=tuple(workspace.memory_candidate_drafts),
            llm_context_packages=tuple(workspace.llm_context_packages),
            llm_tool_intents=tuple(workspace.llm_tool_intents),
            root_cause_assessments=root_cause_assessments,
        )

    async def resume(
        self,
        run_id: UUID,
        *,
        progress_sink: ResearchProgressSink | None = None,
    ) -> ResearchRunResult:
        """Resume one durable run without re-running initial Planning."""

        return await self.run(
            "resume",
            progress_sink=progress_sink,
            _resume_run_id=run_id,
        )

    async def apply_optimization_proposal(
        self,
        proposal_id: UUID,
        *,
        requested_by: str,
    ) -> ResearchOptimizationConfigurationResult:
        """Explicitly request governed activation for one committed Proposal."""

        return await self._optimization_configuration_gateway.request_apply(
            proposal_id,
            requested_by=requested_by,
        )

    def _build_optimization_configuration_gateway(
        self,
        trace_sink: TraceSink,
    ) -> ResearchOptimizationConfigurationGateway:
        return ResearchOptimizationConfigurationGateway(
            proposals=self._persistence.optimization_proposal_store,
            configurations=self._persistence.runtime_configuration,
            governance=self._governance,
            reviews=self._reviews,
            issuer=self._issuer,
            operation_executor=self._operation_executor,
            trace_sink=trace_sink,
            apply_checkpoints=self._persistence.create_decision_checkpoint_store(
                DecisionCheckpoint[
                    OptimizationApplyRequest,
                    OptimizationApplyIntent,
                    OptimizationApplyEffect,
                ]
            ),
            rollback_checkpoints=self._persistence.create_decision_checkpoint_store(
                DecisionCheckpoint[
                    OptimizationRollbackRequest,
                    OptimizationRollbackIntent,
                    OptimizationRollbackEffect,
                ]
            ),
            fault_injector=self._decision_fault_injector,
        )

    async def rollback_optimization(
        self,
        source_apply_effect_fingerprint: str,
        *,
        requested_by: str,
    ) -> ResearchOptimizationConfigurationResult:
        """Explicitly request governed restoration of the prior snapshot."""

        return await self._optimization_configuration_gateway.request_rollback(
            source_apply_effect_fingerprint,
            requested_by=requested_by,
        )

    def close(self) -> None:
        """Release the default SQLite composition explicitly."""

        self._persistence.close()

    @property
    def persistence_identity(self) -> ResearchPersistenceIdentity:
        path = self._persistence.database.path
        adapters = (
            "state",
            "graph",
            "trace",
            "decision",
            "context",
            "archive",
            "memory",
            "review",
            "authorization",
            "artifact",
            "recall_bundle",
            "experience",
            "evaluation",
            "decision_feedback",
            "learning_insight",
            "runtime_configuration",
            "auto_adaptation",
        )
        return ResearchPersistenceIdentity(
            database_path=path,
            adapter_database_paths=tuple((name, path) for name in adapters),
        )

    async def _build_task_definition(
        self,
        *,
        company: str,
        task: str,
        run_id: UUID,
        agent_task: AgentTask,
        trace_sink: TraceSink,
        configuration_snapshot: RuntimeConfigurationSnapshot,
    ) -> tuple[
        ResearchTaskDefinition,
        TaskGraphDraft | None,
        GovernanceRecord | None,
        TaskGraphStore | None,
        ResearchMemoryRecallResult | None,
    ]:
        configured_planner = self._cognitive_capabilities.task_planner
        planner = configured_planner or DeterministicResearchPlanningCapability(
            company
        )
        recall_handler = ResearchInitialMemoryRecallHandler(
            memory_store=self._memory_store,
            bundle_store=self._persistence.memory_recall_bundle_store,
            experience_store=self._persistence.experience_metadata_store,
            capability=(
                self._cognitive_capabilities.memory_recall
                or DeterministicMemoryRecallCapability()
            ),
            governance=self._governance,
            reviews=self._reviews,
            issuer=self._issuer,
            operation_executor=self._operation_executor,
            trace_sink=trace_sink,
            checkpoint_store=self._persistence.create_decision_checkpoint_store(
                DecisionCheckpoint[
                    MemoryRecallRequest,
                    MemoryRecallDraft,
                    MemoryRecallEffect,
                ]
            ),
        )
        try:
            recall_result = await recall_handler.recall(
                goal=task,
                run_id=run_id,
                task_id=agent_task.task_id,
                scope=MemoryScope(project_id="research", agent_scope="planner"),
                facts={"domain": "financial_research"},
                tags=("research",),
            )
        except Exception:
            # Recall is optional for Planning. Failure is fail-closed: no
            # candidate or Draft content is injected into Planner context.
            recall_result = None
        planning = await run_adaptive_planning(
            company=company,
            task=task,
            run_id=run_id,
            agent_task=agent_task,
            planner=planner,
            execution_policy=self._planning_execution_policy,
            governance=self._governance,
            reviews=self._reviews,
            issuer=self._issuer,
            operation_executor=self._operation_executor,
            trace_sink=trace_sink,
            graph_store=self._persistence.task_graph_store,
            checkpoint_store=self._persistence.create_decision_checkpoint_store(
                DecisionCheckpoint[
                    PlanningDecisionPayload,
                    TaskGraphDraft,
                    PlanningGraphEffect,
                ]
            ),
            commit_permit_verifier=self._persistence.commit_permit_verifier,
            deferred_news=configured_planner is None,
            recall_bundle=(
                recall_result.bundle if recall_result is not None else None
            ),
            configuration_snapshot=configuration_snapshot,
        )
        return (
            planning.definition,
            planning.draft if configured_planner is not None else None,
            planning.governance_record,
            planning.graph_store,
            recall_result,
        )

    async def _judge_evaluation(
        self,
        evaluation: EvaluationReport,
        workspace: ResearchWorkspace,
        runtime_result: RunResult,
    ) -> JudgeAssessmentDraft | None:
        judge = self._cognitive_capabilities.evaluation_judge
        if judge is None:
            return None
        report_value = workspace.output_for_role(REPORT_GENERATION)
        references = (
            EvidenceReference(
                reference_id="runtime:completion",
                kind="runtime.result",
                summary="Runtime reached its recorded terminal state.",
            ),
            EvidenceReference(
                reference_id="evaluation:deterministic",
                kind="evaluation.report",
                summary="Deterministic evaluators produced the recorded scores.",
            ),
            EvidenceReference(
                reference_id="artifact:report",
                kind="generated.report",
                summary="Runtime accepted the final research report artifact.",
            ),
        )
        turn = await judge.assess(
            JudgeRequest(
                subject={
                    "runtime_result": {
                        "status": runtime_result.final_state.status.value,
                        "step_count": runtime_result.final_state.step_count,
                        "output": runtime_result.final_state.output,
                    },
                    "deterministic_evaluation": evaluation.model_dump(
                        mode="json"
                    ),
                    "report": report_value,
                },
                criteria=(
                    CHINESE_OUTPUT_INSTRUCTION,
                    "The report is supported by the supplied research evidence.",
                    "The execution outcome and trajectory are internally consistent.",
                    "Findings identify uncertainty without inventing facts.",
                ),
                evidence_catalog=references,
            ),
            invocation=CapabilityInvocationMetadata(
                correlation=InferenceCorrelation(
                    run_id=runtime_result.final_state.run_id,
                    task_id=runtime_result.final_state.task.task_id,
                    action_id=uuid4(),
                ),
                trace_attributes={"operation": "evaluation.assess"},
            ),
        )
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Evaluation Judge returned ToolIntent to Runtime")
        if turn.result is None:
            raise RuntimeError("Evaluation Judge returned no assessment")
        return turn.result

    async def run_demo(
        self,
        task: str,
        *,
        progress_sink: ResearchProgressSink | None = None,
    ) -> ResearchRunResult:
        """Run exactly one request; cross-run evidence comes only from SQLite."""

        return await self.run(task, progress_sink=progress_sink)

    async def _seed_memories(
        self,
        goal_context: ContextUnit,
        workspace: ResearchWorkspace,
        context_trace: ContextMemoryTraceAdapter,
    ) -> None:
        candidates = (
            MemoryCandidate(
                memory_key="research.report_preference",
                content={
                    "format": "markdown",
                    "emphasize": ["risks", "financial metrics"],
                },
                condition=MemoryCondition(
                    facts={"domain": "financial_research"},
                    required_tags=("research",),
                    description="Apply to financial research reports.",
                ),
                evidence=(
                    MemoryEvidence(
                        source_context_id=goal_context.context_id,
                        note="The current task requests an investment report.",
                        weight=0.95,
                    ),
                    MemoryEvidence(
                        source_reference="user:report-preference",
                        note="User prefers concise Markdown with explicit risks.",
                        weight=0.95,
                    ),
                ),
                confidence=0.95,
                evolution=MemoryEvolutionType.EXTEND,
                scope=MemoryScope(project_id="research", agent_scope="planner"),
            ),
            MemoryCandidate(
                memory_key="research.analysis_experience",
                content={
                    "principle": (
                        "Compare growth, margins, leverage, industry, and risks."
                    )
                },
                condition=MemoryCondition(
                    facts={"domain": "financial_research"},
                    required_tags=("research",),
                    description="Use for future financial research behavior.",
                ),
                evidence=(
                    MemoryEvidence(
                        source_context_id=goal_context.context_id,
                        note="The research goal requires multi-angle analysis.",
                        weight=0.95,
                    ),
                    MemoryEvidence(
                        source_reference="experience:research-method",
                        note="Prior research supports a multi-factor review.",
                        weight=0.95,
                    ),
                ),
                confidence=0.95,
                evolution=MemoryEvolutionType.EXTEND,
                scope=MemoryScope(project_id="research", agent_scope="planner"),
            ),
        )
        for candidate in candidates:
            request = MemoryGovernanceAdapter().to_request(
                candidate,
                run_id=goal_context.metadata.run_id,
                task_id=goal_context.metadata.task_id,
                history=GovernanceHistory(successful_similar=5),
            )
            record = self._authorize_with_demo_review(
                request,
                scenario="memory",
            )
            workspace.governance_records.append(record)
            if record.authorization is None:
                raise RuntimeError("Memory Candidate was not authorized")
            update = await self._apply_governed_memory(candidate, record)
            workspace.trace_batches.append(
                context_trace.memory_update(
                    update,
                    correlation=EvaluationCorrelation(
                        run_id=goal_context.metadata.run_id,
                        task_id=goal_context.metadata.task_id,
                    ),
                )
            )

    async def _reinforce_memory(
        self,
        goal_context: ContextUnit,
        workspace: ResearchWorkspace,
        context_trace: ContextMemoryTraceAdapter,
    ) -> None:
        memories = await self._memory_store.list_all()
        preference = next(
            memory
            for memory in memories
            if memory.memory_key == "research.report_preference"
        )
        candidate = MemoryCandidate(
            memory_key=preference.memory_key,
            content=preference.content,
            condition=preference.condition,
            evidence=(
                MemoryEvidence(
                    source_context_id=goal_context.context_id,
                    note="A new research run reused the stored report preference.",
                    weight=0.9,
                ),
            ),
            confidence=0.9,
            evolution=MemoryEvolutionType.SUPPORT,
            target_memory_id=preference.memory_id,
            scope=preference.scope,
            sensitivity=preference.sensitivity,
            expires_at=preference.expires_at,
        )
        request = MemoryGovernanceAdapter().to_request(
            candidate,
            run_id=goal_context.metadata.run_id,
            task_id=goal_context.metadata.task_id,
            history=GovernanceHistory(successful_similar=5),
        )
        record = self._authorize_with_demo_review(request, scenario="memory")
        workspace.governance_records.append(record)
        if record.authorization is None:
            raise RuntimeError("Memory evidence update was not authorized")
        update = await self._apply_governed_memory(candidate, record)
        workspace.trace_batches.append(
            context_trace.memory_update(
                update,
                correlation=EvaluationCorrelation(
                    run_id=goal_context.metadata.run_id,
                    task_id=goal_context.metadata.task_id,
                ),
            )
        )

    async def _apply_governed_memory(
        self,
        candidate: MemoryCandidate,
        record: GovernanceRecord,
    ) -> MemoryUpdateResult:
        authorization = record.authorization
        if authorization is None:
            raise RuntimeError("Memory Candidate was not authorized")

        async def consolidate() -> MemoryUpdateResult:
            return await self._memory_consolidator.consolidate(candidate)

        return await self._operation_executor.execute(
            request=record.request,
            decision=record.final,
            authorization=authorization,
            target=BoundGovernedOperation(
                module_id="research_agent.memory_mutation",
                operation=record.request.operation,
                target=record.request.target,
                subject=candidate,
                apply=consolidate,
            ),
        )

    def _authorize_with_demo_review(
        self,
        request: GovernanceRequest,
        *,
        scenario: str,
    ) -> GovernanceRecord:
        preliminary = self._governance.evaluate(request)
        final = preliminary
        review = None
        if preliminary.outcome is DecisionOutcome.REVIEW_REQUIRED:
            review_id = preliminary.review_request_id
            if review_id is None:
                raise RuntimeError("review-required decision has no review id")
            review = self._reviews.resolve(
                review_id,
                HumanReviewDecision(
                    outcome=ReviewOutcome.APPROVE,
                    reviewer_id="research-demo-reviewer",
                    rationale=(
                        "The demo operation is bounded, reversible, and auditable."
                    ),
                    decided_at=datetime.now(timezone.utc),
                ),
            )
            final = self._governance.finalize_review(request, review)
        authorization = (
            self._issuer.issue(request, final)
            if final.outcome is DecisionOutcome.ALLOW
            else None
        )
        return GovernanceRecord(
            scenario=scenario,
            request=request,
            preliminary=preliminary,
            final=final,
            authorization=authorization,
            review=review,
        )
