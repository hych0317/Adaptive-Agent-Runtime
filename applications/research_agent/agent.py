"""Research Agent Application composed exclusively from Runtime public APIs."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

from adaptive_agent_runtime import (
    AgentRuntime,
    AgentTask,
    InMemoryStateStore,
    InMemoryTraceSink,
    RunResult,
    TraceSink,
)
from adaptive_agent_runtime.context_memory import (
    ConditionalMemoryRecall,
    ContextAssembly,
    ContextAssembler,
    ContextCompressor,
    ContextLayer,
    ContextLifecycleManager,
    ContextLifecycleRuntime,
    ContextMemoryCoordinator,
    ContextMetadata,
    ContextRequirement,
    ContextScheduler,
    ContextSource,
    ContextUnit,
    DeterministicContextLifecyclePolicy,
    DeterministicContextPressureMonitor,
    EvidenceDrivenMemoryConsolidator,
    InMemoryContextArchive,
    InMemoryContextStore,
    InMemoryMemoryStore,
    MemoryCandidate,
    MemoryCondition,
    MemoryEvidence,
    MemoryEvolutionType,
    MemoryUpdateResult,
    ResidencyPolicy,
)
from adaptive_agent_runtime.evaluation import (
    AgentEvaluationPipeline,
    ConservativeOptimizationAgent,
    ConservativeOptimizationPolicy,
    ContextMemoryComponentEvaluator,
    ContextMemoryTraceAdapter,
    DeterministicFailureAnalyzer,
    DeterministicOutcomeEvaluator,
    DeterministicTrajectoryEvaluator,
    EvaluationCorrelation,
    EvaluationCriteria,
    EvaluationInputAssembler,
    EvaluationReport,
    EvaluationResult,
    GraphSnapshotAdapter,
    OrchestrationComponentEvaluator,
    RuntimeNativeTraceCollector,
    RecoveryTraceAdapter,
    ToolComponentEvaluator,
)
from adaptive_agent_runtime.governance import (
    BoundGovernedOperation,
    ConfidenceSignals,
    DecisionOutcome,
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernanceCorrelation,
    GovernanceEvidence,
    GovernanceEvaluator,
    GovernanceHistory,
    GovernanceRequest,
    GovernanceScope,
    GovernanceTarget,
    GovernedOperationExecutor,
    HumanReviewDecision,
    InMemoryHumanReviewService,
    InMemoryAuthorizationConsumptionStore,
    MemoryGovernanceAdapter,
    ImpactAssessment,
    OptimizationGovernanceAdapter,
    ReviewOutcome,
    RiskLevel,
    RuntimeGovernanceEvaluator,
    StrictAuthorizationVerifier,
    SUBJECT_FINGERPRINT_ATTRIBUTE,
    governance_fingerprint,
    default_governance_policy,
)
from adaptive_agent_runtime.orchestration import (
    DeterministicFailureDrivenReplanner,
    DynamicTaskGraphPlanner,
    IsolatedAgentExecutor,
    StrategyActionExecutor,
)
from adaptive_agent_runtime.llm import (
    AutonomousAgentBackend,
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    EvidenceReference,
    InferenceCorrelation,
    JudgeAssessmentDraft,
    JudgeRequest,
    MemoryCandidateDraft,
    MemoryExtractionRequest,
    TaskGraphDraft,
    TaskPlanningRequest,
)
from adaptive_agent_runtime.tool_ecosystem import ToolExecutionStrategy

from applications.research_agent.capabilities import (
    CALCULATION,
    DOCUMENT_ANALYSIS,
    INFORMATION_RETRIEVAL,
    build_research_tool_stack,
)
from applications.research_agent.cognition import ResearchCognitiveCapabilities
from applications.research_agent.report import (
    GovernanceRecord,
    ResearchReport,
    ResearchRunResult,
)
from applications.research_agent.progress import (
    ObservableTraceSink,
    ResearchProgressEvent,
    ResearchProgressKind,
    ResearchProgressSink,
    publish_progress,
)
from applications.research_agent.strategies import (
    DeterministicContextCompressor,
    DeterministicRiskAgent,
    GovernedRecordingToolExecutor,
    GovernedContextLifecycleExecutor,
    GovernedToolIntentExecutor,
    ReportStrategy,
    ResearchStrategy,
    ResearchGraphMutationApplier,
    ResearchContextRequirementProvider,
    ResearchRecoveryPlanApplier,
    ResearchWorkspace,
    ReviewStrategy,
    LLMRiskAgent,
    LLMContextCompressor,
    LLMReadyTaskNodeSelector,
    build_tool_execution_strategy,
)
from applications.research_agent.tasks import (
    REPORT_GENERATION,
    REPORT_STRATEGY_ID,
    RESEARCH_STRATEGY_ID,
    ResearchTaskDefinition,
    REVIEW_STRATEGY_ID,
    build_research_task,
    build_research_task_from_draft,
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


class ResearchAgent:
    """Application facade demonstrating the full Adaptive Runtime stack."""

    def __init__(
        self,
        *,
        autonomous_risk_backend: AutonomousAgentBackend | None = None,
        cognitive_capabilities: ResearchCognitiveCapabilities | None = None,
    ) -> None:
        self._cognitive_capabilities = (
            cognitive_capabilities or ResearchCognitiveCapabilities()
        )
        self._tools = build_research_tool_stack()
        self._context_store = InMemoryContextStore()
        self._context_archive = InMemoryContextArchive()
        compressor: ContextCompressor
        self._llm_context_compressor: LLMContextCompressor | None = None
        compression_capability = self._cognitive_capabilities.context_compressor
        if compression_capability is None:
            compressor = DeterministicContextCompressor()
        else:
            compression_context = self._cognitive_capabilities.compression_context
            if compression_context is None:
                raise ValueError(
                    "LLM Context Compressor requires an egress projection"
                )
            llm_compressor = LLMContextCompressor(
                compression_capability,
                compression_context,
            )
            self._llm_context_compressor = llm_compressor
            compressor = llm_compressor
        self._context_lifecycle = ContextLifecycleManager(
            store=self._context_store,
            archive=self._context_archive,
            compressor=compressor,
        )
        self._memory_store = InMemoryMemoryStore()
        self._memory_consolidator = EvidenceDrivenMemoryConsolidator(
            self._memory_store
        )
        self._reviews = InMemoryHumanReviewService()
        self._governance: GovernanceEvaluator = RuntimeGovernanceEvaluator(
            policy=default_governance_policy(),
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=self._reviews,
        )
        self._issuer = GovernanceAuthorizationIssuer()
        self._authorization_store = InMemoryAuthorizationConsumptionStore()
        self._operation_executor = GovernedOperationExecutor(
            verifier=StrictAuthorizationVerifier(),
            consumption_store=self._authorization_store,
        )
        self._evaluation_history: list[EvaluationResult] = []
        self._memory_seeded = False
        self._autonomous_risk_backend = autonomous_risk_backend

    async def run(
        self,
        task: str,
        *,
        progress_sink: ResearchProgressSink | None = None,
    ) -> ResearchRunResult:
        """Execute one research run and evaluate it after Runtime completion."""

        company = company_from_task(task)
        run_id = uuid4()
        await publish_progress(
            progress_sink,
            ResearchProgressEvent(
                run_id=run_id,
                kind=ResearchProgressKind.RUN_PREPARING,
                payload={"task": task, "company": company},
            ),
        )
        agent_task = AgentTask(
            description=task,
            input={"company": company, "application": "research_agent"},
        )
        definition, llm_task_graph_draft, planning_record = (
            await self._build_task_definition(
                company=company,
                task=task,
                run_id=run_id,
                agent_task=agent_task,
            )
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
        context_coordinator = ContextMemoryCoordinator(
            context_store=self._context_store,
            scheduler=ContextScheduler(),
            assembler=ContextAssembler(),
            requirements=ResearchContextRequirementProvider(
                workspace,
                max_units=64,
                max_tokens=8192,
            ),
            memory_recall=ConditionalMemoryRecall(self._memory_store),
            lifecycle=ContextLifecycleRuntime(
                store=self._context_store,
                pressure_monitor=DeterministicContextPressureMonitor(
                    max_resident_tokens=768,
                ),
                policy=DeterministicContextLifecyclePolicy(
                    max_compressed_units=1,
                ),
                executor=GovernedContextLifecycleExecutor(
                    manager=self._context_lifecycle,
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
        context_trace = ContextMemoryTraceAdapter()
        runtime_trace_sink = InMemoryTraceSink()
        runtime_trace_writer: TraceSink = runtime_trace_sink
        if progress_sink is not None:
            runtime_trace_writer = ObservableTraceSink(
                runtime_trace_sink,
                progress_sink,
            )
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
        await self._context_lifecycle.add(goal_context)
        workspace.context_units.append(goal_context)
        workspace.trace_batches.append(
            context_trace.context_unit(goal_context, kind="context.research_goal")
        )
        if not self._memory_seeded:
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
        tool_strategy: ToolExecutionStrategy = build_tool_execution_strategy(
            workspace=workspace,
            executor=governed_executor,
            resolver=self._tools.resolver,
            selector=self._tools.selector,
        )
        research_strategy = ResearchStrategy(
            tool_strategy=tool_strategy,
            workspace=workspace,
            lifecycle=self._context_lifecycle,
            coordinator=context_coordinator,
            context_trace=context_trace,
            reasoner=self._cognitive_capabilities.reasoner,
            mutation_planner=self._cognitive_capabilities.mutation_planner,
            mutation_context=self._cognitive_capabilities.mutation_context,
            authorize_mutation=(
                (
                    lambda request: self._authorize_with_demo_review(
                        request,
                        scenario="graph_mutation",
                    )
                )
                if self._cognitive_capabilities.mutation_planner is not None
                else None
            ),
            tool_intent_executor=(
                GovernedToolIntentExecutor(
                    resolver=self._tools.resolver,
                    selector=self._tools.selector,
                    executor=governed_executor,
                    workspace=workspace,
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
            lifecycle=self._context_lifecycle,
            coordinator=context_coordinator,
            context_trace=context_trace,
            llm_generator=self._cognitive_capabilities.report_generator,
            llm_context=self._cognitive_capabilities.report_context,
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
                lifecycle=self._context_lifecycle,
                coordinator=context_coordinator,
                context_trace=context_trace,
            )
            action_planner = self._cognitive_capabilities.action_planner
            ready_node_selector = (
                LLMReadyTaskNodeSelector(
                    capability=action_planner,
                    workspace=workspace,
                    operation_executor=self._operation_executor,
                    authorize=lambda request: self._authorize_with_demo_review(
                        request,
                        scenario="action_proposal",
                    ),
                )
                if action_planner is not None
                else None
            )
            planner = DynamicTaskGraphPlanner(
                definition.initial_graph,
                ready_node_selector=ready_node_selector,
                mutation_applier=ResearchGraphMutationApplier(
                    authorize=lambda request: self._authorize_with_demo_review(
                        request,
                        scenario="graph_mutation_apply",
                    ),
                    operation_executor=self._operation_executor,
                    workspace=workspace,
                ),
                recovery_planner=DeterministicFailureDrivenReplanner(
                    max_attempts_per_node=2,
                ),
                recovery_applier=ResearchRecoveryPlanApplier(
                    authorize=lambda request: self._authorize_with_demo_review(
                        request,
                        scenario="failure_recovery",
                    ),
                    operation_executor=self._operation_executor,
                    workspace=workspace,
                ),
            )
            runtime = AgentRuntime(
                planner=planner,
                executor=StrategyActionExecutor(
                    (research_strategy, review_strategy, report_strategy)
                ),
                state_store=InMemoryStateStore(),
                trace_sink=runtime_trace_writer,
                max_steps=20,
            )
            runtime_result = await runtime.run(agent_task, run_id=run_id)
        finally:
            if isolated_directory is not None:
                isolated_directory.cleanup()
        final_state = runtime_result.final_state
        final_graph = planner.graph_for(run_id)

        await self._extract_llm_memories(
            workspace=workspace,
            run_id=run_id,
            task_id=agent_task.task_id,
            context_trace=context_trace,
        )
        if self._llm_context_compressor is not None:
            workspace.llm_context_packages.extend(
                self._llm_context_compressor.take_packages(run_id)
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
                planner.recovery_records_for(run_id),
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
        self._evaluation_history.extend(evaluation.results)
        llm_judgement = await self._judge_evaluation(
            evaluation,
            workspace,
            runtime_result,
        )
        failure_analysis = DeterministicFailureAnalyzer().analyze(
            self._evaluation_history
        )
        proposals = ConservativeOptimizationAgent().propose(
            failure_analysis,
            ConservativeOptimizationPolicy(
                min_affected_runs=2,
                min_pattern_confidence=0.7,
                min_expected_benefit=0.5,
                min_evidence_findings=2,
            ),
        )
        for proposal in proposals:
            request = OptimizationGovernanceAdapter().to_request(proposal)
            workspace.governance_records.append(
                self._authorize_with_demo_review(request, scenario="optimization")
            )

        raw_report = workspace.output_for_role(REPORT_GENERATION)
        if raw_report is None:
            raise RuntimeError(
                "Research Runtime produced no report: "
                + (final_state.error or final_state.status.value)
            )
        report = ResearchReport.model_validate(raw_report)
        memories = await self._memory_store.list_all()
        history_runs = len(
            {result.run_id for result in self._evaluation_history}
        )
        authorization_uses_list = []
        for record in workspace.governance_records:
            if record.authorization is None:
                continue
            use = await self._authorization_store.load(
                record.authorization.authorization_id
            )
            if use is not None:
                authorization_uses_list.append(use)
        authorization_uses = tuple(authorization_uses_list)
        return ResearchRunResult(
            runtime_result=runtime_result,
            task_graph=final_graph,
            report=report,
            evaluation=evaluation,
            failure_analysis=failure_analysis,
            optimization_proposals=proposals,
            governance_records=tuple(workspace.governance_records),
            authorization_uses=authorization_uses,
            runtime_trace=runtime_entries,
            tool_trace=tool_entries,
            tool_observations=tuple(workspace.tool_observations),
            context_units=tuple(workspace.context_units),
            context_assemblies=tuple(workspace.context_assemblies),
            memories=memories,
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
            evaluation_history_runs=history_runs,
        )

    async def _extract_llm_memories(
        self,
        *,
        workspace: ResearchWorkspace,
        run_id: UUID,
        task_id: UUID,
        context_trace: ContextMemoryTraceAdapter,
    ) -> None:
        extractor = self._cognitive_capabilities.memory_extractor
        if extractor is None:
            return
        successful = tuple(
            observation
            for observation in workspace.tool_observations
            if observation.succeeded
        )
        if not successful:
            return
        evidence = tuple(
            EvidenceReference(
                reference_id=f"tool:{item.invocation_id}",
                kind="tool.observation",
                summary=(
                    f"Accepted output from capability '{item.capability_id}'."
                ),
            )
            for item in successful
        )
        existing = await self._memory_store.list_all()
        existing_by_reference = {
            str(memory.memory_id): memory for memory in existing
        }
        projection = self._cognitive_capabilities.extraction_context
        if projection is None:
            raise RuntimeError(
                "Memory Extraction requires a Context egress projection"
            )
        successful_ids = {
            str(observation.invocation_id) for observation in successful
        }
        projected_units: list[ContextUnit] = []
        seen_references: set[str] = set()
        for unit in workspace.context_units:
            if unit.metadata.source is not ContextSource.OBSERVATION:
                continue
            content = unit.content
            if not isinstance(content, Mapping):
                continue
            metadata = content.get("metadata")
            tool = metadata.get("tool") if isinstance(metadata, Mapping) else None
            invocation_id = (
                tool.get("invocation_id") if isinstance(tool, Mapping) else None
            )
            if not isinstance(invocation_id, str):
                continue
            if invocation_id not in successful_ids:
                continue
            reference = unit.metadata.source_reference or str(unit.context_id)
            if reference in seen_references:
                continue
            seen_references.add(reference)
            projected_units.append(unit)
        for assembly in workspace.context_assemblies:
            for unit in assembly.units:
                if unit.metadata.source is not ContextSource.MEMORY_RECALL:
                    continue
                reference = unit.metadata.source_reference or str(unit.context_id)
                if reference in seen_references:
                    continue
                seen_references.add(reference)
                projected_units.append(unit)
        if not projected_units:
            raise RuntimeError("Memory Extraction has no Runtime Context evidence")
        working = tuple(
            unit
            for unit in projected_units
            if unit.metadata.layer is ContextLayer.WORKING
        )
        task_units = tuple(
            unit
            for unit in projected_units
            if unit.metadata.layer is ContextLayer.TASK
        )
        semantic = tuple(
            unit
            for unit in projected_units
            if unit.metadata.layer is ContextLayer.SEMANTIC
        )
        ordered_units = (*working, *task_units, *semantic)
        context_tokens = sum(
            unit.metadata.estimated_tokens for unit in ordered_units
        )
        extraction_assembly = ContextAssembly(
            requirement=ContextRequirement(
                run_id=run_id,
                task_id=task_id,
                goal=(
                    "Extract evidence-bound Memory Candidates from approved "
                    "Runtime Context."
                ),
                max_units=len(ordered_units),
                max_tokens=context_tokens,
            ),
            units=ordered_units,
            working_context=working,
            task_context=task_units,
            semantic_context=semantic,
            used_tokens=context_tokens,
        )
        package = projection.project(extraction_assembly)
        workspace.llm_context_packages.append(package)
        observation_context_ids = tuple(
            str(block.context_id)
            for block in package.blocks
            if block.source is ContextSource.OBSERVATION
        )
        memory_context_ids = tuple(
            str(block.context_id)
            for block in package.blocks
            if block.source is ContextSource.MEMORY_RECALL
        )
        turn = await extractor.extract(
            MemoryExtractionRequest(
                observations={
                    "context_package": package.model_dump(mode="json"),
                    "observation_context_ids": list(observation_context_ids),
                },
                evidence_catalog=evidence,
                existing_memories={
                    "memory_context_ids": list(memory_context_ids),
                    "memory_reference_ids": list(existing_by_reference),
                },
                existing_memory_reference_ids=tuple(existing_by_reference),
            ),
            invocation=CapabilityInvocationMetadata(
                correlation=InferenceCorrelation(
                    run_id=run_id,
                    task_id=task_id,
                    action_id=uuid4(),
                ),
                trace_attributes={"operation": "memory.extract"},
            ),
        )
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Memory Extractor returned ToolIntent to Runtime")
        if turn.result is None:
            raise RuntimeError("Memory Extractor returned no candidates")
        for draft in turn.result:
            await self._apply_memory_draft(
                draft=draft,
                evidence_catalog={item.reference_id for item in evidence},
                existing_by_reference=existing_by_reference,
                workspace=workspace,
                run_id=run_id,
                task_id=task_id,
                context_trace=context_trace,
            )

    async def _apply_memory_draft(
        self,
        *,
        draft: MemoryCandidateDraft,
        evidence_catalog: set[str],
        existing_by_reference: Mapping[str, object],
        workspace: ResearchWorkspace,
        run_id: UUID,
        task_id: UUID,
        context_trace: ContextMemoryTraceAdapter,
    ) -> None:
        unknown_evidence = set(draft.evidence_reference_ids) - evidence_catalog
        if unknown_evidence:
            raise RuntimeError("Memory draft cites unknown evidence")
        target_memory_id = None
        if draft.target_memory_reference is not None:
            target = existing_by_reference.get(draft.target_memory_reference)
            if target is None:
                raise RuntimeError("Memory draft targets unknown Memory")
            target_memory_id = UUID(draft.target_memory_reference)
        candidate = MemoryCandidate(
            memory_key=draft.memory_key,
            content=draft.content,
            condition=MemoryCondition(
                facts=draft.condition.facts,
                required_tags=draft.condition.required_tags,
                description=draft.condition.description,
            ),
            evidence=tuple(
                MemoryEvidence(
                    source_reference=reference_id,
                    note="LLM-extracted candidate cites an accepted Tool observation.",
                    weight=draft.confidence,
                )
                for reference_id in draft.evidence_reference_ids
            ),
            confidence=draft.confidence,
            evolution=MemoryEvolutionType(draft.evolution.value),
            target_memory_id=target_memory_id,
        )
        workspace.memory_candidate_drafts.append(draft)
        governance_request = MemoryGovernanceAdapter().to_request(
            candidate,
            run_id=run_id,
            task_id=task_id,
            history=GovernanceHistory(successful_similar=5),
        )
        governance_record = self._authorize_with_demo_review(
            governance_request,
            scenario="llm_memory",
        )
        workspace.governance_records.append(governance_record)
        if governance_record.authorization is None:
            raise RuntimeError("Governance denied LLM Memory Candidate")
        update = await self._apply_governed_memory(candidate, governance_record)
        workspace.trace_batches.append(
            context_trace.memory_update(
                update,
                correlation=EvaluationCorrelation(
                    run_id=run_id,
                    task_id=task_id,
                ),
            )
        )

    async def _build_task_definition(
        self,
        *,
        company: str,
        task: str,
        run_id: UUID,
        agent_task: AgentTask,
    ) -> tuple[
        ResearchTaskDefinition,
        TaskGraphDraft | None,
        GovernanceRecord | None,
    ]:
        planner = self._cognitive_capabilities.task_planner
        if planner is None:
            return build_research_task(company), None, None
        planner_action_id = uuid4()
        turn = await planner.propose(
            TaskPlanningRequest(
                task=task,
                constraints=(
                    "Use exactly the eight documented research node keys.",
                    "Keep news_analysis in the final plan; Runtime adds it dynamically.",
                    "Use only Runtime-supplied strategy identifiers.",
                ),
                available_strategies=(
                    RESEARCH_STRATEGY_ID,
                    REVIEW_STRATEGY_ID,
                    REPORT_STRATEGY_ID,
                ),
                available_execution_capability_ids=(
                    INFORMATION_RETRIEVAL,
                    DOCUMENT_ANALYSIS,
                    CALCULATION,
                ),
            ),
            invocation=CapabilityInvocationMetadata(
                correlation=InferenceCorrelation(
                    run_id=run_id,
                    task_id=agent_task.task_id,
                    action_id=planner_action_id,
                ),
                trace_attributes={"operation": "graph.initialize.propose"},
            ),
        )
        if turn.kind is CapabilityTurnKind.TOOL_INTENT:
            raise RuntimeError("Task Planner returned ToolIntent to Runtime")
        if turn.result is None:
            raise RuntimeError("Task Planner returned no graph draft")
        definition = build_research_task_from_draft(company, turn.result)
        governance_request = GovernanceRequest(
            scope=GovernanceScope.STATE,
            operation="graph.initialize",
            target=GovernanceTarget(
                target_type="task_graph_draft",
                target_id=str(agent_task.task_id),
            ),
            risk=RiskLevel.MEDIUM,
            signals=ConfidenceSignals(
                stated_confidence=0.9,
                evidence=(
                    GovernanceEvidence(
                        evidence_id=f"task:{agent_task.task_id}",
                        kind="user.task",
                        source="conversation",
                        reliability=1.0,
                        summary="The graph proposal originates from the user task.",
                    ),
                    GovernanceEvidence(
                        evidence_id=f"graph-validation:{agent_task.task_id}",
                        kind="graph.validation",
                        source="research_agent",
                        reliability=1.0,
                        summary=(
                            "The proposal passed role, strategy, dependency, and DAG validation."
                        ),
                    ),
                ),
                impact=ImpactAssessment(
                    score=0.4,
                    reversible=True,
                    description=(
                        "The draft initializes only this run's task graph."
                    ),
                ),
                history=GovernanceHistory(successful_similar=5),
            ),
            correlation=GovernanceCorrelation(
                run_id=run_id,
                task_id=agent_task.task_id,
                action_id=planner_action_id,
            ),
            attributes={
                SUBJECT_FINGERPRINT_ATTRIBUTE: governance_fingerprint(
                    turn.result
                ),
                "node_keys": [node.node_key for node in turn.result.nodes],
                "planner_capability_id": planner.capability_id,
            },
        )
        governance_record = self._authorize_with_demo_review(
            governance_request,
            scenario="planning",
        )
        if governance_record.authorization is None:
            raise RuntimeError("Governance denied LLM task graph proposal")

        async def adopt_definition() -> ResearchTaskDefinition:
            return definition

        accepted_definition = await self._operation_executor.execute(
            request=governance_request,
            decision=governance_record.final,
            authorization=governance_record.authorization,
            target=BoundGovernedOperation(
                module_id="research_agent.graph_initialization",
                operation=governance_request.operation,
                target=governance_request.target,
                subject=turn.result,
                apply=adopt_definition,
            ),
        )
        return accepted_definition, turn.result, governance_record

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
        """Run a baseline plus current run so conservative optimization can recur."""

        await self.run(task)
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
        self._memory_seeded = True

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
