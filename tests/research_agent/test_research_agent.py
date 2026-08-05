"""Phase 7 Research Agent acceptance and boundary tests."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

from adaptive_agent_runtime import RunStatus
from adaptive_agent_runtime.context_memory import (
    ContextLifecycleState,
    ContextSource,
)
from adaptive_agent_runtime.evaluation import EvaluationVerdict
from adaptive_agent_runtime.governance import (
    AuthorizationUseStatus,
    DecisionOutcome,
)
from adaptive_agent_runtime.llm import (
    ActionProposalDraft,
    ActionProposalRequest,
    AgentActionKind,
    AgentActionTraceEntry,
    AgentTargetProfile,
    AutonomousAgentRequest,
    AutonomousAgentResult,
    BackendAvailability,
    BackendDelegatedAccess,
    BackendKind,
    BackendProbeResult,
    BackendTransportFeatures,
    CapabilityContextPolicy,
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
    CompressedContextDraft,
    CompressionRequest,
    ContextEgressPolicy,
    ContextSensitivity,
    GeneratedArtifactDraft,
    GenerationRequest,
    GraphMutationOperationDraft,
    GraphMutationOperationKind,
    GraphMutationProposalDraft,
    GraphMutationProposalRequest,
    JudgeAssessmentDraft,
    JudgeFindingDraft,
    JudgeRequest,
    JudgeSeverity,
    InferenceTargetProfile,
    MemoryCandidateDraft,
    MemoryConditionDraft,
    MemoryEvolutionDraft,
    MemoryExtractionRequest,
    ReasoningContext,
    ReasoningResult,
    PolicyEnforcedContextAdapter,
    StructuredOutputLevel,
    TaskGraphDraft,
    TaskNodeDraft,
    TaskPlanningRequest,
    ToolIntentDraft,
)
from adaptive_agent_runtime.orchestration import TaskNode, TaskNodeStatus

from applications.research_agent import (
    ResearchAgent,
    ResearchCognitiveCapabilities,
    ResearchContextProjection,
    ResearchInformationMode,
    ResearchReportContextProjection,
    build_research_task,
    build_research_task_from_draft,
)
from applications.research_agent.capabilities import (
    CALCULATION,
    DOCUMENT_ANALYSIS,
    INFORMATION_RETRIEVAL,
    LLM_COMPANY_PROVIDER,
    LLM_COMPETITOR_PROVIDER,
    LLM_INDUSTRY_PROVIDER,
    LLM_NEWS_PROVIDER,
    REPORT_GENERATION,
)
from applications.research_agent.cli import format_result
from applications.research_agent.prompts import CHINESE_OUTPUT_INSTRUCTION
from applications.research_agent.report import ReportSection, ResearchReport
from applications.research_agent.web import build_research_view
from applications.research_agent.tasks import (
    NEWS_ANALYSIS,
    build_research_mutations_from_draft,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeAutonomousRiskBackend:
    module_id = "test.agent_backend.research_risk"

    def __init__(self) -> None:
        self._profile = AgentTargetProfile(
            target_id="agent/fake-risk",
            backend_id="fake-agent",
            backend_kind=BackendKind.CLI,
            adapter_version="1",
            model_id="fake-risk-model",
            delegated_access=BackendDelegatedAccess(
                filesystem_read=True,
                shell_execution=True,
            ),
        )
        self.requests: list[AutonomousAgentRequest] = []
        self.workspace_existed_during_execution = False

    @property
    def target_id(self) -> str:
        return self._profile.target_id

    @property
    def profile(self) -> AgentTargetProfile:
        return self._profile

    async def probe(self) -> BackendProbeResult:
        return BackendProbeResult(
            target_id=self.target_id,
            availability=BackendAvailability.AVAILABLE,
            runtime_version="fake",
            active_auth_method="fake_session",
            protocol_version="fake-agent-v1",
        )

    async def execute(
        self,
        request: AutonomousAgentRequest,
    ) -> AutonomousAgentResult:
        self.requests.append(request)
        self.workspace_existed_during_execution = Path(
            request.workspace_root
        ).is_dir()
        payload = request.input
        assert isinstance(payload, Mapping)
        return AutonomousAgentResult(
            request_id=request.request_id,
            target_id=self.target_id,
            model_id=self.profile.model_id,
            output={
                "reviewer": "fake-autonomous-risk-agent",
                "independent": True,
                "node_id": str(payload["node_id"]),
                "risks": [
                    "Autonomous review identified a governed concentration risk."
                ],
                "evidence_reviewed": {
                    "financial_metrics": payload["financial_metrics"] is not None,
                    "industry_analysis": payload["industry_analysis"] is not None,
                },
            },
            actions=(
                AgentActionTraceEntry(
                    sequence=1,
                    kind=AgentActionKind.COMMAND_EXECUTION,
                    item_id="fake-action",
                    status="completed",
                ),
            ),
            remote_request_id="fake-agent-run",
        )


class FakeReportGenerator:
    module_id = "test.cognitive.report_generator"
    capability_id = "generation"

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []
        self.invocations: list[CapabilityInvocationMetadata | None] = []

    async def generate(
        self,
        request: GenerationRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[GeneratedArtifactDraft]:
        self.requests.append(request)
        self.invocations.append(invocation)
        context = request.context
        assert isinstance(context, Mapping)
        company = str(context["company"])
        report = ResearchReport(
            company=company,
            executive_summary="LLM-generated summary grounded in Runtime evidence.",
            sections=(
                ReportSection(
                    title="LLM Analysis",
                    findings=("Evidence-bound generated finding.",),
                ),
            ),
            risk_factors=("LLM-generated governed risk.",),
            investment_view="LLM-generated balanced view.",
            markdown=f"# {company} LLM Investment Research",
        )
        return CapabilityTurnResult[GeneratedArtifactDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=GeneratedArtifactDraft(
                media_type=request.media_type,
                content=report.model_dump(mode="json"),
                evidence_reference_ids=tuple(
                    item.reference_id for item in request.evidence
                ),
            ),
        )


class FakeHybridResearchGenerator(FakeReportGenerator):
    async def generate(
        self,
        request: GenerationRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[GeneratedArtifactDraft]:
        context = request.context
        assert isinstance(context, Mapping)
        topic = context.get("research_topic")
        if topic is None:
            return await super().generate(request, invocation=invocation)
        self.requests.append(request)
        self.invocations.append(invocation)
        company = str(context["company"])
        outputs = {
            "company profile": {
                "company": company,
                "business": "LLM synthesized business profile",
                "research_scope": "LLM research scope",
                "key_facts": ["Model-synthesized company fact"],
                "caveats": ["Verify against primary sources"],
            },
            "industry analysis": {
                "company": company,
                "industry": "LLM synthesized industry",
                "outlook": "LLM synthesized outlook",
                "drivers": ["Model-synthesized driver"],
                "caveats": ["Verify against primary sources"],
            },
            "competitor analysis": {
                "company": company,
                "peers": ["LLM Peer"],
                "differentiation": "LLM synthesized differentiation",
                "competitive_risks": ["Model-synthesized risk"],
                "caveats": ["Verify against primary sources"],
            },
            "news and catalyst analysis": {
                "company": company,
                "signal": "uncertain",
                "catalyst": "LLM synthesized catalyst",
                "controversy": "LLM synthesized controversy",
                "as_of": str(context["requested_at"]),
                "caveats": ["No live news retrieval"],
            },
        }
        return CapabilityTurnResult[GeneratedArtifactDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=GeneratedArtifactDraft(
                media_type=request.media_type,
                content=outputs[str(topic)],
            ),
        )


class FakeEvaluationJudge:
    module_id = "test.cognitive.evaluation_judge"
    capability_id = "judge"

    def __init__(self) -> None:
        self.requests: list[JudgeRequest] = []
        self.invocations: list[CapabilityInvocationMetadata | None] = []

    async def assess(
        self,
        request: JudgeRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[JudgeAssessmentDraft]:
        self.requests.append(request)
        self.invocations.append(invocation)
        return CapabilityTurnResult[JudgeAssessmentDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=JudgeAssessmentDraft(
                summary="The generated report is consistent with Runtime evidence.",
                score=0.9,
                findings=(
                    JudgeFindingDraft(
                        code="evidence.consistent",
                        severity=JudgeSeverity.INFO,
                        summary="Evidence references are present.",
                        assessment_confidence=0.9,
                        evidence_reference_ids=(
                            request.evidence_catalog[0].reference_id,
                        ),
                    ),
                ),
            ),
        )


def planned_graph_draft(company: str) -> TaskGraphDraft:
    specifications = (
        ("company_research", (), "research"),
        ("financial_document", ("company_research",), "research"),
        ("financial_metrics", ("financial_document",), "research"),
        ("industry_analysis", ("company_research",), "research"),
        ("competitor_analysis", ("company_research",), "research"),
        ("news_analysis", ("company_research",), "research"),
        (
            "risk_review",
            (
                "financial_metrics",
                "industry_analysis",
                "competitor_analysis",
                "news_analysis",
            ),
            "review",
        ),
        ("report_generation", ("risk_review",), "report"),
    )
    return TaskGraphDraft(
        nodes=tuple(
            TaskNodeDraft(
                node_key=role,
                goal=f"LLM plan {role} for {company}",
                dependency_keys=dependencies,
                expected_output=f"Validated {role} output",
                requested_strategy_id=strategy,
            )
            for role, dependencies, strategy in specifications
        ),
        rationale="A bounded research DAG proposal.",
    )


class FakeTaskPlanner:
    module_id = "test.cognitive.task_planner"
    capability_id = "planning"

    def __init__(self) -> None:
        self.requests: list[TaskPlanningRequest] = []
        self.invocations: list[CapabilityInvocationMetadata | None] = []

    async def propose(
        self,
        request: TaskPlanningRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[TaskGraphDraft]:
        self.requests.append(request)
        self.invocations.append(invocation)
        return CapabilityTurnResult[TaskGraphDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=planned_graph_draft("Tesla"),
        )


class FakeActionPlanner:
    module_id = "test.cognitive.action_planner"
    capability_id = "action_proposal"

    def __init__(self) -> None:
        self.requests: list[ActionProposalRequest] = []
        self.invocations: list[CapabilityInvocationMetadata | None] = []

    async def propose_action(
        self,
        request: ActionProposalRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ActionProposalDraft]:
        self.requests.append(request)
        self.invocations.append(invocation)
        selected = request.candidates[-1]
        return CapabilityTurnResult[ActionProposalDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=ActionProposalDraft(
                node_key=selected.node_key,
                rationale="Choose one member of the Runtime ready set.",
                evidence_reference_ids=(request.evidence[0].reference_id,),
            ),
        )


class FakeMutationPlanner:
    module_id = "test.cognitive.mutation_planner"
    capability_id = "graph_mutation_proposal"

    def __init__(self) -> None:
        self.requests: list[GraphMutationProposalRequest] = []
        self.invocations: list[CapabilityInvocationMetadata | None] = []

    async def propose_mutations(
        self,
        request: GraphMutationProposalRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[GraphMutationProposalDraft]:
        self.requests.append(request)
        self.invocations.append(invocation)
        evidence_id = request.evidence[0].reference_id
        return CapabilityTurnResult[GraphMutationProposalDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=GraphMutationProposalDraft(
                operations=(
                    GraphMutationOperationDraft(
                        kind=GraphMutationOperationKind.ADD_NODE,
                        node=TaskNodeDraft(
                            node_key="news_analysis",
                            goal="LLM-proposed bounded news analysis",
                            dependency_keys=("company_research",),
                            expected_output="Material news evidence",
                            requested_strategy_id="research",
                        ),
                        reason="Company evidence requires current news review.",
                        evidence_reference_ids=(evidence_id,),
                    ),
                    GraphMutationOperationDraft(
                        kind=GraphMutationOperationKind.ADD_DEPENDENCY,
                        node_key="risk_review",
                        dependency_key="news_analysis",
                        reason="Risk review must consume the news evidence.",
                        evidence_reference_ids=(evidence_id,),
                    ),
                ),
                rationale="Add only the Runtime-allowlisted news branch.",
            ),
        )


class FakeReasoner:
    module_id = "test.cognitive.reasoner"
    capability_id = "reasoning"

    def __init__(self) -> None:
        self.requests: list[ReasoningContext] = []
        self.invocations: list[CapabilityInvocationMetadata | None] = []

    async def analyze(
        self,
        request: ReasoningContext,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ReasoningResult]:
        self.requests.append(request)
        self.invocations.append(invocation)
        return CapabilityTurnResult[ReasoningResult](
            kind=CapabilityTurnKind.COMPLETED,
            result=ReasoningResult(
                conclusions=("The accepted Tool output supports this analysis.",),
                evidence_reference_ids=(request.evidence[0].reference_id,),
            ),
        )


class FakeToolCallingReasoner:
    module_id = "test.cognitive.tool_calling_reasoner"
    capability_id = "reasoning"

    def __init__(self) -> None:
        self.requests: list[ReasoningContext] = []
        self.invocations: list[CapabilityInvocationMetadata | None] = []

    async def analyze(
        self,
        request: ReasoningContext,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ReasoningResult]:
        self.requests.append(request)
        self.invocations.append(invocation)
        if len(self.requests) == 1:
            return CapabilityTurnResult[ReasoningResult](
                kind=CapabilityTurnKind.TOOL_INTENT,
                tool_intents=(
                    ToolIntentDraft(
                        call_key="retrieve-industry-once",
                        capability_id=INFORMATION_RETRIEVAL,
                        arguments={
                            "company": "Tesla",
                            "scope": "industry",
                            "query": "industry evidence",
                        },
                    ),
                ),
            )
        return CapabilityTurnResult[ReasoningResult](
            kind=CapabilityTurnKind.COMPLETED,
            result=ReasoningResult(
                conclusions=("Runtime-governed evidence supports the analysis.",),
                evidence_reference_ids=tuple(
                    item.reference_id for item in request.evidence
                ),
            ),
        )


class FakeContextCompressor:
    module_id = "test.cognitive.context_compressor"
    capability_id = "compression"

    def __init__(self) -> None:
        self.requests: list[CompressionRequest] = []
        self.invocations: list[CapabilityInvocationMetadata | None] = []

    async def compress(
        self,
        request: CompressionRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[CompressedContextDraft]:
        self.requests.append(request)
        self.invocations.append(invocation)
        return CapabilityTurnResult[CompressedContextDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=CompressedContextDraft(
                content={"llm_summary": "Semantically compressed evidence."},
                core_conclusions=("The accepted evidence was retained.",),
                source_reference_ids=(request.source_reference_id,),
                estimated_tokens=request.target_max_tokens,
            ),
        )


class FakeMemoryExtractor:
    module_id = "test.cognitive.memory_extractor"
    capability_id = "extraction"

    def __init__(self) -> None:
        self.requests: list[MemoryExtractionRequest] = []
        self.invocations: list[CapabilityInvocationMetadata | None] = []

    async def extract(
        self,
        request: MemoryExtractionRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[tuple[MemoryCandidateDraft, ...]]:
        self.requests.append(request)
        self.invocations.append(invocation)
        reference_id = request.evidence_catalog[0].reference_id
        return CapabilityTurnResult[tuple[MemoryCandidateDraft, ...]](
            kind=CapabilityTurnKind.COMPLETED,
            result=(
                MemoryCandidateDraft(
                    memory_key="research.llm_extracted_signal",
                    content={
                        "principle": "Retain evidence-backed operating signals."
                    },
                    condition=MemoryConditionDraft(
                        facts={"domain": "financial_research"},
                        required_tags=("research",),
                    ),
                    evidence_reference_ids=(reference_id,),
                    confidence=0.9,
                    evolution=MemoryEvolutionDraft.EXTEND,
                ),
            ),
        )


def compression_context_projection(
    capability_id: str,
) -> ResearchContextProjection:
    target = InferenceTargetProfile(
        target_id="fake/compression-context",
        backend_id="fake",
        backend_kind=BackendKind.LOCAL,
        adapter_version="1",
        model_id="fake-compression-model",
        features=BackendTransportFeatures(
            structured_output=StructuredOutputLevel.JSON_SCHEMA,
        ),
    )
    return ResearchContextProjection(
        adapter=PolicyEnforcedContextAdapter(),
        target=target,
        capability_policy=CapabilityContextPolicy(
            cognitive_capability_id=capability_id,
            max_context_tokens=8192,
        ),
        egress_policy=ContextEgressPolicy(
            allowed_target_ids=(target.target_id,),
            allowed_sensitivities=(ContextSensitivity.INTERNAL,),
            redact_keys=("authorization", "password", "token"),
            max_context_tokens=8192,
        ),
    )


class ResearchTaskInitializationTests(unittest.TestCase):
    def test_application_owns_provider_neutral_dynamic_task_definition(self) -> None:
        definition = build_research_task("Tesla")
        news = definition.node(NEWS_ANALYSIS)

        self.assertEqual(len(definition.nodes), 8)
        self.assertEqual(len(definition.initial_graph.nodes), 7)
        self.assertNotIn(
            news.node_id,
            {node.node_id for node in definition.initial_graph.nodes},
        )
        mutations = definition.discovery_mutations()
        self.assertEqual(len(mutations), 2)
        self.assertTrue(
            all(
                forbidden not in field.lower()
                for field in TaskNode.model_fields
                for forbidden in (
                    "tool",
                    "provider",
                    "capability",
                    "context",
                    "memory",
                )
            )
        )

    def test_research_mutation_converter_rejects_semantically_wrong_edge(
        self,
    ) -> None:
        definition = build_research_task("Tesla")
        proposal = GraphMutationProposalDraft(
            operations=(
                GraphMutationOperationDraft(
                    kind=GraphMutationOperationKind.ADD_NODE,
                    node=TaskNodeDraft(
                        node_key="news_analysis",
                        goal="Review news",
                        dependency_keys=("company_research",),
                        expected_output="News evidence",
                        requested_strategy_id="research",
                    ),
                    reason="News is relevant.",
                    evidence_reference_ids=("context:company",),
                ),
                GraphMutationOperationDraft(
                    kind=GraphMutationOperationKind.ADD_DEPENDENCY,
                    node_key="report_generation",
                    dependency_key="news_analysis",
                    reason="Try to bypass risk review.",
                    evidence_reference_ids=("context:company",),
                ),
            ),
            rationale="Invalid Research edge.",
        )

        with self.assertRaisesRegex(ValueError, "risk_review"):
            build_research_mutations_from_draft(definition, proposal)
        self.assertEqual(len(definition.initial_graph.nodes), 7)

    def test_llm_graph_draft_is_domain_validated_before_runtime_identity(self) -> None:
        draft = planned_graph_draft("Tesla")
        definition = build_research_task_from_draft("Tesla", draft)

        self.assertEqual(len(definition.nodes), 8)
        self.assertEqual(len(definition.initial_graph.nodes), 7)
        self.assertIn("LLM plan company_research", definition.node("company_research").goal)
        risk = definition.node("risk_review")
        news = definition.node("news_analysis")
        self.assertNotIn(news.node_id, risk.dependencies)
        with self.assertRaisesRegex(ValueError, "roles differ"):
            build_research_task_from_draft(
                "Tesla",
                TaskGraphDraft(nodes=draft.nodes[:-1]),
            )
        invalid_report = draft.nodes[-1].model_copy(
            update={"requested_strategy_id": "research"}
        )
        with self.assertRaisesRegex(ValueError, "requires strategy 'report'"):
            build_research_task_from_draft(
                "Tesla",
                TaskGraphDraft(nodes=(*draft.nodes[:-1], invalid_report)),
            )


class ResearchAgentFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_sqlite_resume_reuses_plan_effect_and_report_receipt(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "research.sqlite3"
            first_planner = FakeTaskPlanner()
            first_agent = ResearchAgent(
                persistence_path=path,
                cognitive_capabilities=ResearchCognitiveCapabilities(
                    task_planner=first_planner
                ),
            )
            first = await first_agent.run("分析 Tesla 投资价值")
            run_id = first.runtime_result.final_state.run_id
            self.assertEqual(len(first_planner.requests), 1)
            first_agent.close()

            resumed_planner = FakeTaskPlanner()
            resumed_agent = ResearchAgent(
                persistence_path=path,
                cognitive_capabilities=ResearchCognitiveCapabilities(
                    task_planner=resumed_planner
                ),
            )
            resumed = await resumed_agent.resume(run_id)
            resumed_agent.close()

            self.assertEqual(resumed.runtime_result.final_state.status, RunStatus.COMPLETED)
            self.assertEqual(resumed_planner.requests, [])
            self.assertEqual(
                resumed.report_commit_receipt.effect_fingerprint,
                first.report_commit_receipt.effect_fingerprint,
            )
            self.assertEqual(resumed.report, first.report)

    async def test_full_research_flow_uses_runtime_capabilities_context_and_memory(
        self,
    ) -> None:
        result = await ResearchAgent().run("分析 Tesla 投资价值")

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(len(result.task_graph.nodes), 8)
        self.assertTrue(
            all(node.status is TaskNodeStatus.COMPLETED for node in result.task_graph.nodes)
        )
        news = next(
            node for node in result.task_graph.nodes if "news" in node.goal.lower()
        )
        risk = next(
            node for node in result.task_graph.nodes if "risks" in node.goal.lower()
        )
        self.assertIn(news.node_id, risk.dependencies)

        capabilities = {
            observation.capability_id for observation in result.tool_observations
        }
        self.assertEqual(
            capabilities,
            {
                INFORMATION_RETRIEVAL,
                DOCUMENT_ANALYSIS,
                CALCULATION,
                REPORT_GENERATION,
            },
        )
        self.assertTrue(any(not item.succeeded for item in result.tool_observations))
        self.assertTrue(result.tool_observations[-1].succeeded)

        lifecycle_states = {unit.lifecycle_state for unit in result.context_units}
        self.assertTrue(
            {
                ContextLifecycleState.ACTIVE,
                ContextLifecycleState.COMPRESSED,
                ContextLifecycleState.ARCHIVED,
            }.issubset(lifecycle_states)
        )
        self.assertEqual(len(result.context_assemblies), 8)
        self.assertTrue(
            any(assembly.semantic_context for assembly in result.context_assemblies)
        )
        self.assertTrue(
            any(
                unit.metadata.source is ContextSource.OBSERVATION
                for unit in result.context_units
            )
        )
        self.assertTrue(
            {
                "research.report_preference",
                "research.analysis_experience",
            }.issubset({memory.memory_key for memory in result.memories}),
        )
        self.assertIn("Tesla Investment Research", result.report.markdown)
        self.assertIn("risks", result.report.executive_summary.lower())
        self.assertEqual(result.agent_executions, ())
        self.assertIsNone(result.llm_judgement)
        self.assertIsNone(result.llm_task_graph_draft)
        self.assertEqual(result.llm_action_proposals, ())
        self.assertEqual(result.llm_graph_mutation_proposals, ())
        self.assertEqual(result.llm_reasoning, ())
        self.assertEqual(result.llm_memory_candidates, ())
        self.assertEqual(result.llm_context_packages, ())
        self.assertEqual(result.llm_tool_intents, ())

    async def test_evaluation_and_both_review_scenarios_are_generated(self) -> None:
        result = await ResearchAgent().run_demo("分析 Tesla 投资价值")

        self.assertEqual(result.evaluation.outcome.verdict, EvaluationVerdict.PASS)
        self.assertIsNotNone(result.evaluation.trajectory.score)
        component_names = {
            item.component.value
            for item in result.evaluation.components
            if item.component is not None
        }
        self.assertEqual(
            component_names,
            {"orchestration", "tool", "context_memory"},
        )
        self.assertTrue(result.failure_analysis.patterns)
        self.assertTrue(result.optimization_proposals)
        self.assertGreaterEqual(result.evaluation_history_runs, 2)

        tool_review = next(
            record
            for record in result.governance_records
            if record.scenario == "tool"
            and record.preliminary.outcome is DecisionOutcome.REVIEW_REQUIRED
        )
        optimization_review = next(
            record
            for record in result.governance_records
            if record.scenario == "optimization"
        )
        for record in (tool_review, optimization_review):
            self.assertEqual(record.final.outcome, DecisionOutcome.ALLOW)
            self.assertIsNotNone(record.review)
            self.assertIsNotNone(record.authorization)

    async def test_isolated_risk_result_and_cli_sections_are_visible(self) -> None:
        result = await ResearchAgent().run("分析 Example Corp 投资价值")
        risk = next(
            node
            for node in result.task_graph.nodes
            if node.strategy_id == "review"
        )
        assert risk.observation is not None
        output = risk.observation.output
        self.assertIsInstance(output, Mapping)
        assert isinstance(output, Mapping)
        self.assertEqual(output["independent"], True)
        rendered = format_result(result)
        for heading in (
            "1. Task Graph",
            "2. Execution Trace",
            "3. Research Report",
            "4. Evaluation Report",
            "5. Governance Decisions",
        ):
            self.assertIn(heading, rendered)

    async def test_opt_in_autonomous_risk_backend_is_governed_and_isolated(
        self,
    ) -> None:
        backend = FakeAutonomousRiskBackend()
        result = await ResearchAgent(
            autonomous_risk_backend=backend
        ).run("分析 Example Corp 投资价值")

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(len(backend.requests), 1)
        self.assertTrue(backend.workspace_existed_during_execution)
        self.assertFalse(Path(backend.requests[0].workspace_root).exists())
        delegated = backend.requests[0].policy.delegated_access
        self.assertTrue(delegated.filesystem_read)
        self.assertTrue(delegated.shell_execution)
        self.assertFalse(delegated.filesystem_write)
        self.assertFalse(delegated.mcp_execution)
        self.assertFalse(delegated.arbitrary_network)
        self.assertEqual(len(result.agent_executions), 1)
        self.assertEqual(
            result.agent_executions[0].actions[0].kind,
            AgentActionKind.COMMAND_EXECUTION,
        )
        self.assertIn(
            "governed concentration risk",
            result.report.risk_factors[0],
        )
        agent_governance = next(
            record
            for record in result.governance_records
            if record.scenario == "agent"
        )
        self.assertEqual(agent_governance.request.operation, "agent.execute")
        self.assertEqual(agent_governance.final.outcome, DecisionOutcome.ALLOW)
        self.assertIsNotNone(agent_governance.authorization)

    async def test_opt_in_generation_and_judge_capabilities_return_drafts(
        self,
    ) -> None:
        generator = FakeReportGenerator()
        judge = FakeEvaluationJudge()
        result = await ResearchAgent(
            cognitive_capabilities=ResearchCognitiveCapabilities(
                report_generator=generator,
                evaluation_judge=judge,
            )
        ).run("分析 Tesla 投资价值")

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(len(generator.requests), 1)
        self.assertEqual(len(judge.requests), 1)
        self.assertIn("LLM Investment Research", result.report.markdown)
        self.assertNotIn(
            REPORT_GENERATION,
            {item.capability_id for item in result.tool_observations},
        )
        self.assertIsNotNone(result.llm_judgement)
        assert result.llm_judgement is not None
        self.assertEqual(result.llm_judgement.score, 0.9)
        run_id = result.runtime_result.final_state.run_id
        task_id = result.runtime_result.final_state.task.task_id
        generator_invocation = generator.invocations[0]
        judge_invocation = judge.invocations[0]
        self.assertIsNotNone(generator_invocation)
        self.assertIsNotNone(judge_invocation)
        assert generator_invocation is not None
        assert judge_invocation is not None
        self.assertEqual(generator_invocation.correlation.run_id, run_id)
        self.assertEqual(generator_invocation.correlation.task_id, task_id)
        self.assertIsNotNone(generator_invocation.correlation.node_id)
        self.assertIsNotNone(generator_invocation.correlation.action_id)
        self.assertEqual(
            generator_invocation.trace_attributes["operation"],
            "report.generate",
        )
        self.assertEqual(judge_invocation.correlation.run_id, run_id)
        self.assertEqual(judge_invocation.correlation.task_id, task_id)
        self.assertIsNotNone(judge_invocation.correlation.action_id)
        self.assertEqual(
            judge_invocation.trace_attributes["operation"],
            "evaluation.assess",
        )
        evidence_ids = {
            item.reference_id for item in judge.requests[0].evidence_catalog
        }
        self.assertIn(
            result.llm_judgement.findings[0].evidence_reference_ids[0],
            evidence_ids,
        )
        rendered = format_result(result)
        self.assertIn("LLM Judge:", rendered)

    async def test_llm_research_mode_uses_llm_information_providers(self) -> None:
        generator = FakeHybridResearchGenerator()

        result = await ResearchAgent(
            cognitive_capabilities=ResearchCognitiveCapabilities(
                report_generator=generator,
            ),
            information_mode=ResearchInformationMode.LLM_RESEARCH,
        ).run("分析 Tesla 投资价值")

        information = {
            item.provider_id: item
            for item in result.tool_observations
            if item.capability_id == INFORMATION_RETRIEVAL
        }
        self.assertEqual(
            set(information),
            {
                LLM_COMPANY_PROVIDER,
                LLM_INDUSTRY_PROVIDER,
                LLM_COMPETITOR_PROVIDER,
                LLM_NEWS_PROVIDER,
            },
        )
        for observation in information.values():
            self.assertIsInstance(observation.output, Mapping)
            assert isinstance(observation.output, Mapping)
            self.assertEqual(
                observation.output["source"],
                "llm model synthesis (not live retrieval)",
            )
        self.assertEqual(len(generator.requests), 5)
        self.assertTrue(
            all(
                CHINESE_OUTPUT_INSTRUCTION in request.instruction
                for request in generator.requests
            )
        )
        self.assertIn("LLM Investment Research", result.report.markdown)

    async def test_report_context_is_projected_and_redacted_before_egress(
        self,
    ) -> None:
        generator = FakeReportGenerator()
        target = InferenceTargetProfile(
            target_id="fake/context-target",
            backend_id="fake",
            backend_kind=BackendKind.LOCAL,
            adapter_version="1",
            model_id="fake-context-model",
            features=BackendTransportFeatures(
                structured_output=StructuredOutputLevel.JSON_SCHEMA,
            ),
        )
        projection = ResearchReportContextProjection(
            adapter=PolicyEnforcedContextAdapter(),
            target=target,
            capability_policy=CapabilityContextPolicy(
                cognitive_capability_id=generator.capability_id,
                max_context_tokens=8192,
            ),
            egress_policy=ContextEgressPolicy(
                allowed_target_ids=(target.target_id,),
                allowed_sensitivities=(
                    ContextSensitivity.INTERNAL,
                    ContextSensitivity.CONFIDENTIAL,
                ),
                redact_keys=("format",),
                max_context_tokens=8192,
            ),
        )

        result = await ResearchAgent(
            cognitive_capabilities=ResearchCognitiveCapabilities(
                report_generator=generator,
                report_context=projection,
            )
        ).run("分析 Tesla 投资价值")

        self.assertEqual(len(result.llm_context_packages), 1)
        package = result.llm_context_packages[0]
        self.assertEqual(package.target_id, target.target_id)
        self.assertTrue(package.blocks)
        self.assertTrue(
            any("format" in block.redacted_keys for block in package.blocks)
        )
        request_context = generator.requests[0].context
        assert isinstance(request_context, Mapping)
        self.assertIn("runtime_context", request_context)
        self.assertNotIn("preferences", request_context)

    async def test_planner_proposal_is_governed_and_reasoning_is_supplemental(
        self,
    ) -> None:
        planner = FakeTaskPlanner()
        reasoner = FakeReasoner()
        result = await ResearchAgent(
            cognitive_capabilities=ResearchCognitiveCapabilities(
                task_planner=planner,
                reasoner=reasoner,
            )
        ).run("分析 Tesla 投资价值")

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(len(planner.requests), 1)
        self.assertIsNotNone(result.llm_task_graph_draft)
        self.assertTrue(
            any(
                node.goal.startswith("LLM plan company_research")
                for node in result.task_graph.nodes
            )
        )
        planning_governance = next(
            record
            for record in result.governance_records
            if record.scenario == "planning"
        )
        self.assertEqual(
            planning_governance.request.operation,
            "graph.initialize",
        )
        self.assertEqual(
            planning_governance.request.target.target_type,
            "runtime_run_graph",
        )
        self.assertIn(
            "decision_effect_fingerprint",
            planning_governance.request.attributes,
        )
        self.assertNotEqual(
            planning_governance.request.target.target_type,
            "task_graph_draft",
        )
        self.assertEqual(planning_governance.final.outcome, DecisionOutcome.ALLOW)
        self.assertIsNotNone(planning_governance.authorization)
        planning_invocation = planner.invocations[0]
        self.assertIsNotNone(planning_invocation)
        assert planning_invocation is not None
        self.assertEqual(
            planning_invocation.correlation.action_id,
            planning_governance.request.correlation.action_id,
        )
        self.assertEqual(
            planning_invocation.trace_attributes["operation"],
            "graph.initialize.propose",
        )
        self.assertEqual(
            planning_invocation.trace_attributes["planning_max_agent_calls"],
            1,
        )
        self.assertEqual(
            planning_invocation.trace_attributes["planning_max_retries"],
            0,
        )
        decision_trace_kinds = {
            entry.event.kind
            for entry in result.runtime_trace
            if entry.event.kind.startswith("decision.")
        }
        self.assertTrue(
            {
                "decision.requested",
                "decision.context_projected",
                "decision.proposed",
                "decision.validation_passed",
                "decision.governance_requested",
                "decision.apply_started",
                "decision.applied",
            }.issubset(decision_trace_kinds)
        )
        self.assertEqual(len(reasoner.requests), 6)
        self.assertEqual(len(result.llm_reasoning), 6)
        self.assertTrue(
            all(record.result.evidence_reference_ids for record in result.llm_reasoning)
        )

    async def test_action_and_mutation_proposals_are_context_bounded_and_governed(
        self,
    ) -> None:
        action_planner = FakeActionPlanner()
        mutation_planner = FakeMutationPlanner()
        reasoner = FakeToolCallingReasoner()
        result = await ResearchAgent(
            cognitive_capabilities=ResearchCognitiveCapabilities(
                action_planner=action_planner,
                mutation_planner=mutation_planner,
                reasoner=reasoner,
                reasoner_tool_intent_limit=1,
                mutation_context=compression_context_projection(
                    mutation_planner.capability_id
                ),
            )
        ).run("分析 Tesla 投资价值")

        self.assertEqual(
            result.runtime_result.final_state.status,
            RunStatus.COMPLETED,
        )
        self.assertGreaterEqual(len(action_planner.requests), 1)
        self.assertEqual(len(mutation_planner.requests), 1)
        self.assertEqual(
            len(result.llm_action_proposals),
            len(action_planner.requests),
        )
        self.assertEqual(len(result.llm_graph_mutation_proposals), 1)
        news = next(
            node
            for node in result.task_graph.nodes
            if node.goal == "LLM-proposed bounded news analysis"
        )
        self.assertEqual(news.status, TaskNodeStatus.COMPLETED)
        mutation_request = mutation_planner.requests[0]
        trigger = mutation_request.trigger_observation
        self.assertIsInstance(trigger, Mapping)
        assert isinstance(trigger, Mapping)
        self.assertIn("context_package", trigger)
        self.assertNotIn("company", trigger)
        context_package = trigger["context_package"]
        self.assertIsInstance(context_package, Mapping)
        assert isinstance(context_package, Mapping)
        blocks = context_package["blocks"]
        self.assertIsInstance(blocks, (list, tuple))
        assert isinstance(blocks, (list, tuple))
        self.assertGreaterEqual(len(blocks), 2)
        self.assertTrue(
            any(
                package.cognitive_capability_id
                == mutation_planner.capability_id
                for package in result.llm_context_packages
            )
        )
        action_governance = tuple(
            record
            for record in result.governance_records
            if record.scenario == "action_proposal"
        )
        self.assertEqual(len(action_governance), len(action_planner.requests))
        self.assertTrue(
            all(record.authorization is not None for record in action_governance)
        )
        mutation_governance = next(
            record
            for record in result.governance_records
            if record.scenario == "graph_mutation"
        )
        self.assertEqual(mutation_governance.request.operation, "graph.mutate")
        self.assertIsNotNone(mutation_governance.authorization)
        action_invocations = tuple(
            invocation
            for invocation in action_planner.invocations
            if invocation is not None
        )
        self.assertEqual(
            {
                invocation.correlation.action_id
                for invocation in action_invocations
            },
            {
                record.request.correlation.action_id
                for record in action_governance
            },
        )
        self.assertTrue(
            all(
                invocation.trace_attributes["operation"]
                == "node.select.propose"
                for invocation in action_invocations
            )
        )
        mutation_invocation = mutation_planner.invocations[0]
        self.assertIsNotNone(mutation_invocation)
        assert mutation_invocation is not None
        self.assertEqual(
            mutation_invocation.correlation.action_id,
            mutation_governance.request.correlation.action_id,
        )
        self.assertEqual(
            mutation_invocation.trace_attributes["operation"],
            "graph.mutate.propose",
        )
        consolidated_records = (*action_governance, mutation_governance)
        lifecycle_prefix = (
            "decision.requested",
            "decision.context_projected",
            "decision.proposed",
            "decision.validation_passed",
            "decision.governance_requested",
        )
        for record in consolidated_records:
            attributes = record.request.attributes
            self.assertIn("decision_request_id", attributes)
            self.assertIn("decision_effect_fingerprint", attributes)
            request_id = attributes["decision_request_id"]
            observed = tuple(
                entry.event.kind
                for entry in result.runtime_trace
                if entry.event.kind.startswith("decision.")
                and entry.event.payload["correlation"]["request_id"]
                == request_id
            )
            review_kinds = (
                ("decision.review_required",)
                if record.review is not None
                else ()
            )
            self.assertEqual(
                observed,
                (
                    *lifecycle_prefix,
                    *review_kinds,
                    "decision.authorized",
                    "decision.apply_started",
                    "decision.applied",
                ),
            )
        self.assertFalse(
            any(
                record.scenario == "graph_mutation_apply"
                for record in result.governance_records
            ),
            "Agent mutation must apply the same governed batch, not a second path",
        )

    async def test_reasoner_tool_intent_returns_through_runtime_tool_governance(
        self,
    ) -> None:
        reasoner = FakeToolCallingReasoner()
        result = await ResearchAgent(
            cognitive_capabilities=ResearchCognitiveCapabilities(
                reasoner=reasoner,
                reasoner_tool_intent_limit=1,
            )
        ).run("分析 Tesla 投资价值")

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(len(result.llm_tool_intents), 1)
        record = result.llm_tool_intents[0]
        self.assertEqual(record.intent.capability_id, INFORMATION_RETRIEVAL)
        self.assertTrue(record.observation.succeeded)
        self.assertEqual(len(reasoner.requests), 7)
        followup_evidence = {
            item.reference_id for item in reasoner.requests[1].evidence
        }
        self.assertIn("tool-intent:retrieve-industry-once", followup_evidence)
        invocation_governance = tuple(
            item
            for item in result.governance_records
            if item.scenario == "tool_invocation_decision"
            and item.request.correlation.action_id
            == record.observation.correlation.action_id
        )
        self.assertEqual(len(invocation_governance), 1)
        governance = invocation_governance[0]
        self.assertEqual(governance.request.operation, "tool.call")
        self.assertEqual(governance.final.outcome, DecisionOutcome.ALLOW)
        self.assertIsNotNone(governance.authorization)
        self.assertIn("decision_effect_fingerprint", governance.request.attributes)
        decision_request_id = governance.request.attributes["decision_request_id"]
        decision_trace = tuple(
            entry
            for entry in result.runtime_trace
            if entry.event.kind.startswith("decision.")
            and entry.event.payload["correlation"]["request_id"]
            == decision_request_id
        )
        self.assertEqual(
            tuple(entry.event.kind for entry in decision_trace),
            (
                "decision.requested",
                "decision.context_projected",
                "decision.proposed",
                "decision.validation_passed",
                "decision.governance_requested",
                "decision.authorized",
                "decision.apply_started",
                "decision.applied",
            ),
        )
        projected_context = decision_trace[1].event.payload["decision"]
        self.assertEqual(projected_context["agent_scope"], "tool_invocation")
        self.assertEqual(projected_context["included_source_count"], 1)
        self.assertNotIn("provider_credentials", projected_context)
        self.assertFalse(
            any(
                item.scenario == "tool"
                and item.request.correlation.action_id
                == record.observation.correlation.action_id
                for item in result.governance_records
            ),
            "ToolIntent invocation must not use the legacy manual Tool path",
        )
        self.assertIn(
            CHINESE_OUTPUT_INSTRUCTION,
            reasoner.requests[0].constraints,
        )
        view = build_research_view(
            result,
            task="分析 Tesla 投资价值",
            elapsed_seconds=0.1,
            inference_label="Fake Tool Calling LLM",
            tool_intent_enabled=True,
        )
        projected = view["llm"]["toolIntentRecords"][0]
        self.assertEqual(projected["callKey"], "retrieve-industry-once")
        self.assertEqual(projected["provider"], record.observation.provider_id)
        self.assertEqual(projected["status"], "succeeded")

    async def test_compression_and_memory_extraction_are_governed_drafts(
        self,
    ) -> None:
        compressor = FakeContextCompressor()
        extractor = FakeMemoryExtractor()
        result = await ResearchAgent(
            cognitive_capabilities=ResearchCognitiveCapabilities(
                context_compressor=compressor,
                memory_extractor=extractor,
                compression_context=compression_context_projection(
                    compressor.capability_id
                ),
                extraction_context=compression_context_projection(
                    extractor.capability_id
                ),
            )
        ).run("分析 Tesla 投资价值")

        self.assertEqual(result.runtime_result.final_state.status, RunStatus.COMPLETED)
        self.assertGreaterEqual(len(compressor.requests), 1)
        self.assertLess(
            compressor.requests[0].target_max_tokens,
            compressor.requests[0].original_estimated_tokens,
        )
        compression_payload = compressor.requests[0].content
        self.assertIsInstance(compression_payload, Mapping)
        assert isinstance(compression_payload, Mapping)
        self.assertIn("blocks", compression_payload)
        self.assertEqual(
            len(result.llm_context_packages),
            len(compressor.requests) + 1,
        )
        self.assertEqual(
            {
                package.cognitive_capability_id
                for package in result.llm_context_packages
            },
            {compressor.capability_id, extractor.capability_id},
        )
        self.assertTrue(
            any(
                isinstance(unit.content, Mapping)
                and "llm_summary" in unit.content
                for unit in result.context_units
            )
        )
        self.assertEqual(len(extractor.requests), 1)
        compressor_invocation = compressor.invocations[0]
        extractor_invocation = extractor.invocations[0]
        self.assertIsNotNone(compressor_invocation)
        self.assertIsNotNone(extractor_invocation)
        assert compressor_invocation is not None
        assert extractor_invocation is not None
        self.assertEqual(
            compressor_invocation.trace_attributes["operation"],
            "context.compress",
        )
        self.assertIsNotNone(compressor_invocation.correlation.node_id)
        self.assertEqual(
            extractor_invocation.trace_attributes["operation"],
            "memory.extract",
        )
        self.assertIsNone(extractor_invocation.correlation.node_id)
        extraction_observations = extractor.requests[0].observations
        self.assertIsInstance(extraction_observations, Mapping)
        assert isinstance(extraction_observations, Mapping)
        self.assertIn("context_package", extraction_observations)
        extraction_memories = extractor.requests[0].existing_memories
        self.assertIsInstance(extraction_memories, Mapping)
        assert isinstance(extraction_memories, Mapping)
        self.assertNotIn("content", extraction_memories)
        self.assertEqual(len(result.llm_memory_candidates), 1)
        self.assertIn(
            "research.llm_extracted_signal",
            {memory.memory_key for memory in result.memories},
        )
        memory_governance = next(
            record
            for record in result.governance_records
            if record.scenario == "llm_memory"
        )
        self.assertEqual(memory_governance.request.operation, "memory.write")
        self.assertEqual(memory_governance.final.outcome, DecisionOutcome.ALLOW)
        self.assertIsNotNone(memory_governance.authorization)
        self.assertIn(
            "decision_effect_fingerprint",
            memory_governance.request.attributes,
        )
        memory_request_id = memory_governance.request.attributes[
            "decision_request_id"
        ]
        self.assertEqual(
            tuple(
                entry.event.kind
                for entry in result.runtime_trace
                if entry.event.kind.startswith("decision.")
                and entry.event.payload["correlation"]["request_id"]
                == memory_request_id
            ),
            (
                "decision.requested",
                "decision.context_projected",
                "decision.proposed",
                "decision.validation_passed",
                "decision.governance_requested",
                "decision.review_required",
                "decision.authorized",
                "decision.apply_started",
                "decision.applied",
            ),
        )
        extracted_memory = next(
            memory
            for memory in result.memories
            if memory.memory_key == "research.llm_extracted_signal"
        )
        self.assertTrue(
            all(item.source_reference for item in extracted_memory.evidence)
        )

    async def test_adaptive_changes_consume_authorization_at_apply_point(
        self,
    ) -> None:
        result = await ResearchAgent().run("分析 Tesla 投资价值")

        self.assertTrue(result.authorization_uses)
        self.assertTrue(
            all(
                use.status is AuthorizationUseStatus.APPLIED
                for use in result.authorization_uses
            )
        )
        applied_operations = {use.operation for use in result.authorization_uses}
        self.assertTrue(
            {
                "tool.call",
                "graph.mutate",
                "memory.write",
                "context.compress",
                "context.archive",
                "context.restore",
            }.issubset(applied_operations)
        )


class ApplicationBoundaryTests(unittest.TestCase):
    def test_runtime_has_no_reverse_dependency_on_applications(self) -> None:
        runtime_root = PROJECT_ROOT / "src" / "adaptive_agent_runtime"
        offenders: list[str] = []
        for path in runtime_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    names = (node.module or "",)
                else:
                    continue
                if any(name == "applications" or name.startswith("applications.") for name in names):
                    offenders.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(offenders, [])

    def test_application_imports_runtime_through_public_package_exports(self) -> None:
        application_root = PROJECT_ROOT / "applications" / "research_agent"
        private_imports: list[str] = []
        public_roots = {
            "adaptive_agent_runtime",
            "adaptive_agent_runtime.context_memory",
            "adaptive_agent_runtime.decisioning",
            "adaptive_agent_runtime.evaluation",
            "adaptive_agent_runtime.governance",
            "adaptive_agent_runtime.llm",
            "adaptive_agent_runtime.orchestration",
            "adaptive_agent_runtime.persistence",
            "adaptive_agent_runtime.tool_ecosystem",
        }
        for path in application_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = node.module or ""
                if module.startswith("adaptive_agent_runtime") and module not in public_roots:
                    private_imports.append(f"{path.name}:{module}")
        self.assertEqual(private_imports, [])


if __name__ == "__main__":
    unittest.main()
