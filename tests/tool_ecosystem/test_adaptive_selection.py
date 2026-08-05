from __future__ import annotations

import unittest
from collections.abc import Callable
from uuid import UUID, uuid4

from pydantic import ValidationError

from adaptive_agent_runtime import (
    AgentRuntime,
    AgentState,
    AgentTask,
    InMemoryStateStore,
    InMemoryTraceSink,
    RunStatus,
)
from adaptive_agent_runtime.governance import (
    DeterministicConfidenceEvaluator,
    DeterministicRuleEvaluator,
    GovernanceAuthorizationIssuer,
    GovernedOperationExecutor,
    InMemoryAuthorizationConsumptionStore,
    InMemoryHumanReviewService,
    RuntimeGovernanceEvaluator,
    StrictAuthorizationVerifier,
    default_governance_policy,
)
from adaptive_agent_runtime.llm import (
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    CapabilityTurnResult,
    ToolSelectionDraft,
    ToolSelectionProposalRequest,
    ToolIntentDraft,
)
from adaptive_agent_runtime.orchestration import (
    DeterministicFailureDrivenReplanner,
    DynamicTaskGraph,
    DynamicTaskGraphPlanner,
    RecoveryPlan,
    StrategyActionExecutor,
    TaskNode,
    apply_recovery_plan,
)
from adaptive_agent_runtime.tool_ecosystem import (
    Capability,
    CapabilityRequest,
    CapabilityRequirement,
    CapabilityResolver,
    ExactCapabilityMatcher,
    InMemoryCapabilityCatalog,
    InMemoryToolRegistry,
    InMemoryToolTraceSink,
    ManagedToolExecutor,
    MappedTaskCapabilityRequestProvider,
    ProviderAvailability,
    RetryPolicy,
    ToolExecutionPolicy,
    ToolExecutionStrategy,
    ToolInvocation,
    ToolProviderMetadata,
    ToolProviderResult,
    ToolSelectionExecutionPolicy,
)
from applications.research_agent.report import GovernanceRecord
from applications.research_agent.strategies import ResearchWorkspace
from applications.research_agent.tasks import build_research_task
from applications.research_agent.tool_selection import (
    ResearchToolSelectionDecisionHandler,
)


CAPABILITY_ID = "test.lookup"
SECRET = "provider-secret-must-not-egress"


class RecordingProvider:
    def __init__(
        self,
        provider_id: str,
        *,
        result: ToolProviderResult,
    ) -> None:
        self.provider_id = provider_id
        self.credential = SECRET
        self._result = result
        self.calls: list[ToolInvocation] = []

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        self.calls.append(invocation)
        return self._result


class FakeToolSelectionAgent:
    module_id = "test.tool_selection.agent"
    capability_id = "tool_selection_proposal"

    def __init__(
        self,
        choose: Callable[[ToolSelectionProposalRequest, int], str],
        *,
        after_request: Callable[[], None] | None = None,
        confidence: float = 0.8,
    ) -> None:
        self._choose = choose
        self._after_request = after_request
        self._confidence = confidence
        self.requests: list[ToolSelectionProposalRequest] = []

    async def propose_tool_selection(
        self,
        request: ToolSelectionProposalRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ToolSelectionDraft]:
        del invocation
        self.requests.append(request)
        selected = self._choose(request, len(self.requests))
        if self._after_request is not None:
            self._after_request()
        return CapabilityTurnResult[ToolSelectionDraft](
            kind=CapabilityTurnKind.COMPLETED,
            result=ToolSelectionDraft(
                selected_candidate_ref=selected,
                rationale="Selected the best semantic fit from Runtime candidates.",
                confidence=self._confidence,
            ),
        )


class ToolIntentSelectionAgent:
    module_id = "test.tool_selection.tool_intent_agent"
    capability_id = "tool_selection_proposal"

    async def propose_tool_selection(
        self,
        request: ToolSelectionProposalRequest,
        *,
        invocation: CapabilityInvocationMetadata | None = None,
    ) -> CapabilityTurnResult[ToolSelectionDraft]:
        del request, invocation
        return CapabilityTurnResult[ToolSelectionDraft](
            kind=CapabilityTurnKind.TOOL_INTENT,
            tool_intents=(
                ToolIntentDraft(
                    call_key="escape",
                    capability_id=CAPABILITY_ID,
                    arguments={"query": "bypass Runtime"},
                ),
            ),
        )


class DirectRecoveryApplier:
    module_id = "test.tool_selection.recovery_applier"

    async def apply(
        self,
        graph: DynamicTaskGraph,
        plan: RecoveryPlan,
        *,
        state: AgentState,
    ) -> DynamicTaskGraph:
        del state
        return apply_recovery_plan(graph, plan)


def _metadata(provider_id: str, name: str, *, priority: int) -> ToolProviderMetadata:
    return ToolProviderMetadata(
        provider_id=provider_id,
        name=name,
        capability_id=CAPABILITY_ID,
        description=f"Safe description for {name}",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        tags=("research",),
        selection_priority=priority,
    )


class AdaptiveToolSelectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.catalog = InMemoryCapabilityCatalog()
        self.catalog.register(
            Capability(
                capability_id=CAPABILITY_ID,
                name="Lookup",
                description="Look up bounded evidence.",
                tags=("research",),
            )
        )
        self.registry = InMemoryToolRegistry(self.catalog)
        self.primary = RecordingProvider(
            "provider.primary",
            result=ToolProviderResult.ok(output={"provider": "primary"}),
        )
        self.secondary = RecordingProvider(
            "provider.secondary",
            result=ToolProviderResult.ok(output={"provider": "secondary"}),
        )
        self.registry.register(
            _metadata("provider.primary", "Primary", priority=100),
            self.primary,
        )
        self.registry.register(
            _metadata("provider.secondary", "Secondary", priority=1),
            self.secondary,
        )
        self.resolver = CapabilityResolver(
            catalog=self.catalog,
            registry=self.registry,
            matcher=ExactCapabilityMatcher(),
        )
        self.trace = InMemoryTraceSink()
        self.workspace = ResearchWorkspace(build_research_task("Acme"))
        self.reviews = InMemoryHumanReviewService()
        self.governance = RuntimeGovernanceEvaluator(
            policy=default_governance_policy(),
            rule_evaluator=DeterministicRuleEvaluator(),
            confidence_evaluator=DeterministicConfidenceEvaluator(),
            review_service=self.reviews,
        )
        self.issuer = GovernanceAuthorizationIssuer()
        self.operation_executor = GovernedOperationExecutor(
            verifier=StrictAuthorizationVerifier(),
            consumption_store=InMemoryAuthorizationConsumptionStore(),
        )

    def _request(self) -> CapabilityRequest:
        return CapabilityRequest(
            requirement=CapabilityRequirement(
                capability_id=CAPABILITY_ID,
                required_provider_tags=("research",),
            ),
            arguments={"query": "Acme"},
        )

    def _handler(
        self,
        agent: FakeToolSelectionAgent | ToolIntentSelectionAgent,
    ) -> ResearchToolSelectionDecisionHandler:
        return ResearchToolSelectionDecisionHandler(
            capability=agent,
            resolver=self.resolver,
            execution_policy=ToolSelectionExecutionPolicy(timeout_seconds=2.0),
            governance=self.governance,
            reviews=self.reviews,
            issuer=self.issuer,
            operation_executor=self.operation_executor,
            trace_sink=self.trace,
            workspace=self.workspace,
        )

    def _strategy(
        self,
        node: TaskNode,
        agent: FakeToolSelectionAgent | ToolIntentSelectionAgent,
        *,
        executor: ManagedToolExecutor | None = None,
    ) -> ToolExecutionStrategy:
        return ToolExecutionStrategy(
            requests=MappedTaskCapabilityRequestProvider({node.node_id: self._request()}),
            resolver=self.resolver,
            selection_handler=self._handler(agent),
            executor=(
                executor
                or ManagedToolExecutor(
                    registry=self.registry,
                    trace_sink=InMemoryToolTraceSink(),
                )
            ),
            policy=ToolExecutionPolicy(
                timeout_seconds=1.0,
                retry=RetryPolicy(max_retries=0),
            ),
            strategy_id="tool",
        )

    async def test_agent_selects_only_runtime_candidate_and_cannot_see_secret(
        self,
    ) -> None:
        node = TaskNode(
            goal="prefer the secondary semantic source",
            expected_output="lookup",
            strategy_id="tool",
        )
        agent = FakeToolSelectionAgent(
            lambda request, _: next(
                item.candidate_ref
                for item in request.candidates
                if item.name == "Secondary"
            ),
            confidence=1.0,
        )
        state = AgentState(run_id=uuid4(), task=AgentTask(description="select source"))
        result = await self._strategy(node, agent).execute(node, state)

        self.assertTrue(result.succeeded)
        self.assertEqual(result.output, {"provider": "secondary"})
        self.assertEqual(len(self.primary.calls), 0)
        self.assertEqual(len(self.secondary.calls), 1)
        projected = agent.requests[0].model_dump_json()
        self.assertNotIn(SECRET, projected)
        self.assertNotIn("credential", projected)
        record: GovernanceRecord = self.workspace.governance_records[-1]
        self.assertEqual(record.request.operation, "tool.selection.bind")
        self.assertEqual(record.request.risk.value, "low")
        kinds = tuple(entry.event.kind for entry in self.trace.entries_for(state.run_id))
        self.assertEqual(
            kinds,
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

    async def test_outside_candidate_is_rejected_before_executor(self) -> None:
        node = TaskNode(
            goal="escape candidates",
            expected_output="lookup",
            strategy_id="tool",
        )
        agent = FakeToolSelectionAgent(lambda request, _: "outside-candidate")

        result = await self._strategy(node, agent).execute(
            node,
            AgentState(run_id=uuid4(), task=AgentTask(description="reject escape")),
        )

        self.assertFalse(result.succeeded)
        self.assertIn("did not apply", result.error or "")
        self.assertEqual(len(self.primary.calls) + len(self.secondary.calls), 0)

    async def test_stale_availability_rejects_selection_before_execution(self) -> None:
        node = TaskNode(
            goal="stale candidate",
            expected_output="lookup",
            strategy_id="tool",
        )

        def make_stale() -> None:
            self.registry.set_availability(
                "provider.secondary",
                ProviderAvailability.UNAVAILABLE,
            )

        agent = FakeToolSelectionAgent(
            lambda request, _: next(
                item.candidate_ref
                for item in request.candidates
                if item.name == "Secondary"
            ),
            after_request=make_stale,
        )
        result = await self._strategy(node, agent).execute(
            node,
            AgentState(run_id=uuid4(), task=AgentTask(description="stale selection")),
        )

        self.assertFalse(result.succeeded)
        self.assertIn("did not apply", result.error or "")
        self.assertEqual(len(self.primary.calls) + len(self.secondary.calls), 0)

    async def test_tool_intent_cannot_bypass_selection_apply(self) -> None:
        node = TaskNode(
            goal="attempt ToolIntent escape",
            expected_output="lookup",
            strategy_id="tool",
        )
        result = await self._strategy(node, ToolIntentSelectionAgent()).execute(
            node,
            AgentState(run_id=uuid4(), task=AgentTask(description="reject ToolIntent")),
        )

        self.assertFalse(result.succeeded)
        self.assertIn("cannot execute ToolIntents", result.error or "")
        self.assertEqual(len(self.primary.calls) + len(self.secondary.calls), 0)

    async def test_tool_failure_still_enters_recovery_and_reselects(self) -> None:
        self.primary._result = ToolProviderResult.failed(
            error="temporary timeout",
            retryable=False,
        )
        node = TaskNode(
            goal="recover failed provider",
            expected_output="lookup",
            strategy_id="tool",
        )

        def select_by_attempt(request: ToolSelectionProposalRequest, call: int) -> str:
            name = "Primary" if call == 1 else "Secondary"
            return next(
                item.candidate_ref for item in request.candidates if item.name == name
            )

        agent = FakeToolSelectionAgent(select_by_attempt)
        strategy = self._strategy(node, agent)
        planner = DynamicTaskGraphPlanner(
            DynamicTaskGraph(nodes=(node,)),
            recovery_planner=DeterministicFailureDrivenReplanner(
                max_attempts_per_node=2
            ),
            recovery_applier=DirectRecoveryApplier(),
        )
        result = await AgentRuntime(
            planner=planner,
            executor=StrategyActionExecutor((strategy,)),
            state_store=InMemoryStateStore(),
            trace_sink=self.trace,
        ).run(AgentTask(description="recover adaptive Tool selection"))

        self.assertEqual(result.final_state.status, RunStatus.COMPLETED)
        self.assertEqual(len(self.primary.calls), 1)
        self.assertEqual(len(self.secondary.calls), 1)
        self.assertEqual(len(agent.requests), 2)
        self.assertEqual(len(planner.recovery_records_for(result.final_state.run_id)), 1)

    def test_selection_draft_cannot_contain_execution_fields(self) -> None:
        with self.assertRaises(ValidationError):
            ToolSelectionDraft.model_validate(
                {
                    "selected_candidate_ref": "candidate-1",
                    "rationale": "attempt execution",
                    "confidence": 0.5,
                    "execute_tool": {"provider_id": "provider.primary"},
                }
            )
