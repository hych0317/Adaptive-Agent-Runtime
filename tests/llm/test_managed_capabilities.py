from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any, cast
from uuid import uuid4

from adaptive_agent_runtime.llm import (
    ActionProposalCapability,
    ActionProposalRequest,
    ArtifactGenerationCapability,
    BackendAvailability,
    BackendKind,
    BackendProbeResult,
    BackendTransportFeatures,
    CapabilityDraftValidator,
    CapabilityExecutionPolicyError,
    CapabilityInferenceSettings,
    CapabilityInvocationMetadata,
    CapabilityResultValidationError,
    CapabilityTurnKind,
    CompressionRequest,
    DeterministicInferenceRouter,
    EvidenceJudgeCapability,
    EvidenceReference,
    GatewayArtifactGenerationCapability,
    GatewayActionProposalCapability,
    GatewayEvidenceJudgeCapability,
    GatewayMemoryExtractionCapability,
    GatewayGraphMutationProposalCapability,
    GatewayReasoningCapability,
    GatewayRecoveryProposalCapability,
    GatewayRootCauseAnalysisCapability,
    GatewaySemanticCompressionCapability,
    GatewayTaskGraphProposalCapability,
    GatewayToolSelectionProposalCapability,
    GenerationRequest,
    GraphMutationProposalCapability,
    GraphMutationProposalRequest,
    InferenceGatewayPolicy,
    InferenceCorrelation,
    InferenceRequest,
    InferenceTargetProfile,
    InMemoryInferenceBackendRegistry,
    InMemoryInferenceGatewayTrace,
    JudgeRequest,
    ManagedInferenceGateway,
    MemoryExtractionCapability,
    MemoryExtractionRequest,
    ModelResponseKind,
    NormalizedFinishReason,
    NormalizedModelResponse,
    PlanningActionCandidate,
    ProviderNeutralResponseValidator,
    ReasoningCapability,
    ReasoningContext,
    RecoveryActionDraftKind,
    RecoveryNodeCandidate,
    RecoveryProposalCapability,
    RecoveryProposalRequest,
    RootCauseAnalysisCapability,
    RootCauseAnalysisRequest,
    ResponseSchemaValidationError,
    SemanticCompressionCapability,
    StructuredOutputLevel,
    TaskGraphProposalCapability,
    TaskPlanningRequest,
    ToolIntentDraft,
    ToolIntentMode,
    ToolSpecification,
    ToolSelectionCandidate,
    ToolSelectionProposalCapability,
    ToolSelectionProposalRequest,
)


def evidence(reference_id: str) -> EvidenceReference:
    return EvidenceReference(reference_id=reference_id, kind="test")


class CapabilityResponseBackend:
    module_id = "test.capability_response_backend"

    def __init__(
        self,
        outputs: Mapping[str, Any],
        *,
        tool_intent_capabilities: tuple[str, ...] = (),
    ) -> None:
        self._profile = InferenceTargetProfile(
            target_id="fake/capabilities",
            backend_id="fake",
            backend_kind=BackendKind.LOCAL,
            adapter_version="1",
            model_id="fake-capability-model",
            features=BackendTransportFeatures(
                structured_output=StructuredOutputLevel.JSON_SCHEMA,
                tool_intent=True,
            ),
        )
        self._outputs = dict(outputs)
        self._tool_intent_capabilities = set(tool_intent_capabilities)
        self.requests: list[InferenceRequest] = []

    @property
    def target_id(self) -> str:
        return self._profile.target_id

    @property
    def profile(self) -> InferenceTargetProfile:
        return self._profile

    async def probe(self) -> BackendProbeResult:
        return BackendProbeResult(
            target_id=self.target_id,
            availability=BackendAvailability.AVAILABLE,
        )

    async def invoke(
        self,
        request: InferenceRequest,
    ) -> NormalizedModelResponse:
        self.requests.append(request)
        if request.cognitive_capability_id in self._tool_intent_capabilities:
            return NormalizedModelResponse(
                request_id=request.request_id,
                target_id=self.target_id,
                model_id=self.profile.model_id,
                kind=ModelResponseKind.TOOL_INTENT,
                tool_intents=(
                    ToolIntentDraft(
                        call_key="tool-1",
                        capability_id="search",
                        arguments={"query": "evidence"},
                    ),
                ),
                finish_reason=NormalizedFinishReason.TOOL_INTENT,
            )
        return NormalizedModelResponse(
            request_id=request.request_id,
            target_id=self.target_id,
            model_id=self.profile.model_id,
            kind=ModelResponseKind.OUTPUT,
            output=self._outputs[request.cognitive_capability_id],
            finish_reason=NormalizedFinishReason.COMPLETED,
        )


def gateway_with(
    backend: CapabilityResponseBackend,
) -> ManagedInferenceGateway:
    registry = InMemoryInferenceBackendRegistry()
    registry.register(backend)
    return ManagedInferenceGateway(
        registry=registry,
        router=DeterministicInferenceRouter(),
        response_validator=ProviderNeutralResponseValidator(),
        trace_sink=InMemoryInferenceGatewayTrace(),
    )


class ManagedCapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_selection_completes_through_managed_gateway(self) -> None:
        backend = CapabilityResponseBackend(
            {
                "tool_selection_proposal": {
                    "selected_candidate_ref": "candidate:b",
                    "rationale": "Provider B best matches the semantic request.",
                    "evidence_reference_ids": [],
                    "confidence": 0.9,
                }
            }
        )
        capability = GatewayToolSelectionProposalCapability(
            gateway=gateway_with(backend),
            draft_validator=CapabilityDraftValidator(),
        )

        turn = await capability.propose_tool_selection(
            ToolSelectionProposalRequest(
                task="Research Acme",
                node_goal="Retrieve evidence",
                capability_id="lookup",
                candidates=(
                    ToolSelectionCandidate(
                        candidate_ref="candidate:a",
                        name="Provider A",
                        description="General source",
                        capability_id="lookup",
                    ),
                    ToolSelectionCandidate(
                        candidate_ref="candidate:b",
                        name="Provider B",
                        description="Specialized source",
                        capability_id="lookup",
                    ),
                ),
            )
        )

        self.assertEqual(turn.kind, CapabilityTurnKind.COMPLETED)
        self.assertIsInstance(capability, ToolSelectionProposalCapability)
        self.assertEqual(
            turn.result.selected_candidate_ref if turn.result is not None else None,
            "candidate:b",
        )

    async def test_root_cause_analysis_completes_through_managed_gateway(
        self,
    ) -> None:
        backend = CapabilityResponseBackend(
            {
                "root_cause_analysis": {
                    "conclusion": "supported",
                    "primary": {
                        "code": "source.input_invalid",
                        "description": "The source returned invalid input.",
                        "supporting_evidence_reference_ids": ["failure:1"],
                        "counter_evidence_reference_ids": [],
                    },
                    "alternatives": [],
                    "assumptions": [],
                    "unresolved_questions": [],
                    "rationale": "The direct failure evidence is explicit.",
                    "confidence": 0.8,
                }
            }
        )
        capability = GatewayRootCauseAnalysisCapability(
            gateway=gateway_with(backend),
            draft_validator=CapabilityDraftValidator(),
        )

        turn = await capability.analyze_root_cause(
            RootCauseAnalysisRequest(
                trigger="inline_failure",
                failure_summary="source input is invalid",
                trace_completeness="partial",
                evidence_catalog=(evidence("failure:1"),),
                constraints=("Cite only supplied evidence.",),
            )
        )

        self.assertEqual(turn.kind, CapabilityTurnKind.COMPLETED)
        self.assertIsInstance(capability, RootCauseAnalysisCapability)
        self.assertEqual(
            turn.result.primary.code
            if turn.result is not None and turn.result.primary is not None
            else None,
            "source.input_invalid",
        )

    async def test_recovery_proposal_completes_through_managed_gateway(self) -> None:
        backend = CapabilityResponseBackend(
            {
                "recovery_proposal": {
                    "failure_kind": "transient",
                    "hypothesis": "The upstream failure is transient.",
                    "alternatives": ["Abort"],
                    "selected_action": {
                        "kind": "retry_node",
                        "target_node_ref": "node:failed",
                        "reason": "Retry once.",
                    },
                    "rationale": "A retry has the smallest impact.",
                    "evidence_reference_ids": ["failure:1"],
                    "confidence": 0.8,
                }
            }
        )
        capability = GatewayRecoveryProposalCapability(
            gateway=gateway_with(backend),
            draft_validator=CapabilityDraftValidator(),
        )

        turn = await capability.propose_recovery(
            RecoveryProposalRequest(
                task="Recover the run",
                failed_node_ref="node:failed",
                failed_goal="Collect evidence",
                error="temporary timeout",
                graph_nodes=(
                    RecoveryNodeCandidate(
                        node_ref="node:failed",
                        goal="Collect evidence",
                        status="failed",
                        strategy_id="research",
                    ),
                ),
                available_strategies=("research",),
                allowed_recovery_actions=(RecoveryActionDraftKind.RETRY_NODE,),
                prior_attempts=0,
                max_attempts=1,
                evidence=(evidence("failure:1"),),
            )
        )

        self.assertEqual(turn.kind, CapabilityTurnKind.COMPLETED)
        self.assertIsInstance(capability, RecoveryProposalCapability)
        self.assertEqual(
            turn.result.hypothesis if turn.result is not None else None,
            "The upstream failure is transient.",
        )

    async def test_capability_propagates_required_target_binding(self) -> None:
        backend = CapabilityResponseBackend(
            {
                "reasoning": {
                    "conclusions": ["Bound"],
                    "evidence_reference_ids": [],
                }
            }
        )
        capability = GatewayReasoningCapability(
            gateway=gateway_with(backend),
            draft_validator=CapabilityDraftValidator(),
            settings=CapabilityInferenceSettings(
                required_target_id=backend.target_id,
            ),
        )

        await capability.analyze(ReasoningContext(goal="Use bound target"))

        self.assertEqual(
            backend.requests[0].required_target_id,
            backend.target_id,
        )

    async def test_invocation_metadata_reaches_gateway_request(self) -> None:
        backend = CapabilityResponseBackend(
            {
                "reasoning": {
                    "conclusions": ["Correlated"],
                    "evidence_reference_ids": [],
                }
            }
        )
        capability = GatewayReasoningCapability(
            gateway=gateway_with(backend),
            draft_validator=CapabilityDraftValidator(),
            settings=CapabilityInferenceSettings(
                trace_attributes={"application": "research_agent"},
            ),
        )
        correlation = InferenceCorrelation(
            run_id=uuid4(),
            task_id=uuid4(),
            node_id=uuid4(),
            action_id=uuid4(),
        )

        await capability.analyze(
            ReasoningContext(goal="Trace this turn"),
            invocation=CapabilityInvocationMetadata(
                correlation=correlation,
                trace_attributes={"operation": "node.reason"},
            ),
        )

        captured = backend.requests[0]
        self.assertEqual(captured.correlation, correlation)
        self.assertEqual(
            captured.trace_attributes,
            {
                "application": "research_agent",
                "operation": "node.reason",
            },
        )

    async def test_invocation_cannot_override_deployment_trace_identity(
        self,
    ) -> None:
        backend = CapabilityResponseBackend(
            {
                "reasoning": {
                    "conclusions": ["unused"],
                    "evidence_reference_ids": [],
                }
            }
        )
        capability = GatewayReasoningCapability(
            gateway=gateway_with(backend),
            draft_validator=CapabilityDraftValidator(),
            settings=CapabilityInferenceSettings(
                trace_attributes={"application": "research_agent"},
            ),
        )

        with self.assertRaises(CapabilityExecutionPolicyError):
            await capability.analyze(
                ReasoningContext(goal="Do not relabel deployment"),
                invocation=CapabilityInvocationMetadata(
                    trace_attributes={"application": "another_app"},
                ),
            )
        self.assertEqual(backend.requests, [])

    async def test_all_six_capabilities_complete_through_gateway(self) -> None:
        outputs: dict[str, Any] = {
            "reasoning": {
                "conclusions": ["Supported"],
                "evidence_reference_ids": ["evidence:1"],
            },
            "task_graph_proposal": {
                "nodes": [
                    {
                        "node_key": "research",
                        "goal": "Research",
                        "expected_output": "Evidence",
                        "requested_strategy_id": "research",
                    }
                ]
            },
            "artifact_generation": {
                "media_type": "text/markdown",
                "content": "# Report",
                "evidence_reference_ids": ["evidence:1"],
                "warnings": [],
            },
            "evidence_judge": {
                "summary": "Grounded",
                "score": 1.0,
                "findings": [],
            },
            "semantic_compression": {
                "content": "Short",
                "core_conclusions": ["Conclusion"],
                "source_reference_ids": ["context:1"],
                "estimated_tokens": 20,
            },
            "memory_extraction": {
                "candidates": [
                    {
                        "memory_key": "format.preference",
                        "content": {"format": "markdown"},
                        "condition": {},
                        "evidence_reference_ids": ["observation:1"],
                        "confidence": 0.9,
                        "evolution": "extend",
                    }
                ]
            },
        }
        backend = CapabilityResponseBackend(outputs)
        gateway = gateway_with(backend)
        validator = CapabilityDraftValidator()
        reasoning = GatewayReasoningCapability(
            gateway=gateway,
            draft_validator=validator,
        )
        planning = GatewayTaskGraphProposalCapability(
            gateway=gateway,
            draft_validator=validator,
        )
        generation = GatewayArtifactGenerationCapability(
            gateway=gateway,
            draft_validator=validator,
        )
        judge = GatewayEvidenceJudgeCapability(
            gateway=gateway,
            draft_validator=validator,
        )
        compression = GatewaySemanticCompressionCapability(
            gateway=gateway,
            draft_validator=validator,
        )
        extraction = GatewayMemoryExtractionCapability(
            gateway=gateway,
            draft_validator=validator,
        )

        reasoning_turn = await reasoning.analyze(
            ReasoningContext(
                goal="Analyze",
                evidence=(evidence("evidence:1"),),
            )
        )
        planning_turn = await planning.propose(
            TaskPlanningRequest(
                task="Research",
                available_strategies=("research",),
            )
        )
        generation_turn = await generation.generate(
            GenerationRequest(
                instruction="Write",
                media_type="text/markdown",
                evidence=(evidence("evidence:1"),),
            )
        )
        judge_turn = await judge.assess(
            JudgeRequest(
                subject={"status": "complete"},
                criteria=("grounded",),
                evidence_catalog=(evidence("trace:1"),),
            )
        )
        compression_turn = await compression.compress(
            CompressionRequest(
                source_reference_id="context:1",
                content="Long",
                original_estimated_tokens=100,
                target_max_tokens=30,
            )
        )
        extraction_turn = await extraction.extract(
            MemoryExtractionRequest(
                observations={"format": "markdown"},
                evidence_catalog=(evidence("observation:1"),),
            )
        )

        turns = (
            reasoning_turn,
            planning_turn,
            generation_turn,
            judge_turn,
            compression_turn,
            extraction_turn,
        )
        self.assertTrue(
            all(turn.kind is CapabilityTurnKind.COMPLETED for turn in turns)
        )
        self.assertEqual(len(extraction_turn.result or ()), 1)
        self.assertIsInstance(reasoning, ReasoningCapability)
        self.assertIsInstance(planning, TaskGraphProposalCapability)
        self.assertIsInstance(generation, ArtifactGenerationCapability)
        self.assertIsInstance(judge, EvidenceJudgeCapability)
        self.assertIsInstance(compression, SemanticCompressionCapability)
        self.assertIsInstance(extraction, MemoryExtractionCapability)
        self.assertEqual(
            tuple(request.cognitive_capability_id for request in backend.requests),
            tuple(outputs),
        )
        first_input = cast(Mapping[str, Any], backend.requests[0].input)
        self.assertEqual(first_input["contract_version"], "1")
        self.assertIn("payload", first_input)

    async def test_tool_intent_is_returned_without_execution(self) -> None:
        backend = CapabilityResponseBackend(
            {},
            tool_intent_capabilities=("reasoning",),
        )
        capability = GatewayReasoningCapability(
            gateway=gateway_with(backend),
            draft_validator=CapabilityDraftValidator(),
            settings=CapabilityInferenceSettings(
                tool_intent=ToolIntentMode.ALLOWED,
                eligible_tools=(
                    ToolSpecification(
                        capability_id="search",
                        name="Search",
                        description="Search evidence",
                    ),
                ),
            ),
        )

        turn = await capability.analyze(ReasoningContext(goal="Research"))

        self.assertEqual(turn.kind, CapabilityTurnKind.TOOL_INTENT)
        self.assertIsNone(turn.result)
        self.assertEqual(turn.tool_intents[0].capability_id, "search")
        self.assertEqual(len(backend.requests), 1)

    async def test_generation_embeds_runtime_artifact_schema_for_strict_output(
        self,
    ) -> None:
        backend = CapabilityResponseBackend(
            {
                "artifact_generation": {
                    "media_type": "application/json",
                    "content": {
                        "title": "Report",
                        "sections": [{"finding": "Grounded"}],
                    },
                    "evidence_reference_ids": [],
                    "warnings": [],
                }
            }
        )
        capability = GatewayArtifactGenerationCapability(
            gateway=gateway_with(backend),
            draft_validator=CapabilityDraftValidator(),
        )

        turn = await capability.generate(
            GenerationRequest(
                instruction="Generate a report",
                media_type="application/json",
                output_schema={
                    "$defs": {
                        "Section": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "finding": {"type": "string"},
                            },
                            "required": ["finding"],
                        }
                    },
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "sections": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Section"},
                        },
                    },
                    "required": ["title", "sections"],
                },
            )
        )

        self.assertIsNotNone(turn.result)
        captured_schema = backend.requests[0].response_schema
        self.assertIsNotNone(captured_schema)
        assert captured_schema is not None
        self.assertEqual(captured_schema["type"], "object")
        required = cast(list[Any], captured_schema["required"])
        self.assertEqual(
            set(required),
            {
                "media_type",
                "content",
                "evidence_reference_ids",
                "warnings",
            },
        )
        definitions = captured_schema["$defs"]
        self.assertIsInstance(definitions, Mapping)
        assert isinstance(definitions, Mapping)
        self.assertNotIn("JsonValue", definitions)
        self.assertIn("Section", definitions)
        properties = cast(Mapping[str, Any], captured_schema["properties"])
        self.assertIsInstance(properties, Mapping)
        assert isinstance(properties, Mapping)
        media_type = cast(Mapping[str, Any], properties["media_type"])
        self.assertEqual(media_type["const"], "application/json")
        content = cast(Mapping[str, Any], properties["content"])
        self.assertIsInstance(content, Mapping)
        assert isinstance(content, Mapping)
        self.assertEqual(content["type"], "object")
        content_properties = cast(Mapping[str, Any], content["properties"])
        sections = cast(Mapping[str, Any], content_properties["sections"])
        items = cast(Mapping[str, Any], sections["items"])
        self.assertEqual(items["$ref"], "#/$defs/Section")
        self.assertNotIn("default", str(captured_schema))

    async def test_planning_action_and_mutation_proposals_complete_through_gateway(
        self,
    ) -> None:
        backend = CapabilityResponseBackend(
            {
                "action_proposal": {
                    "node_key": "industry",
                    "rationale": "Use the ready evidence branch.",
                    "evidence_reference_ids": ["runtime:ready"],
                },
                "graph_mutation_proposal": {
                    "operations": [
                        {
                            "kind": "add_node",
                            "node": {
                                "node_key": "news",
                                "goal": "Review news",
                                "dependency_keys": ["company"],
                                "expected_output": "News evidence",
                                "requested_strategy_id": "research",
                            },
                            "reason": "Company evidence requires news review.",
                            "evidence_reference_ids": ["observation:company"],
                        },
                        {
                            "kind": "add_dependency",
                            "node_key": "risk",
                            "dependency_key": "news",
                            "reason": "Risk review needs news evidence.",
                            "evidence_reference_ids": ["observation:company"],
                        },
                    ],
                    "rationale": "Extend the graph with bounded news analysis.",
                },
            }
        )
        validator = CapabilityDraftValidator()
        action = GatewayActionProposalCapability(
            gateway=gateway_with(backend),
            draft_validator=validator,
        )
        mutation = GatewayGraphMutationProposalCapability(
            gateway=gateway_with(backend),
            draft_validator=validator,
        )

        action_turn = await action.propose_action(
            ActionProposalRequest(
                task="Research Tesla",
                candidates=(
                    PlanningActionCandidate(
                        node_key="industry",
                        goal="Analyze industry",
                        expected_output="Industry evidence",
                        strategy_id="research",
                    ),
                ),
                evidence=(evidence("runtime:ready"),),
            )
        )
        mutation_turn = await mutation.propose_mutations(
            GraphMutationProposalRequest(
                task="Research Tesla",
                trigger_node_key="company",
                trigger_observation={"company": "Tesla"},
                existing_node_keys=("company", "risk"),
                allowed_new_node_keys=("news",),
                available_strategies=("research",),
                evidence=(evidence("observation:company"),),
            )
        )

        self.assertIs(action_turn.kind, CapabilityTurnKind.COMPLETED)
        self.assertIs(mutation_turn.kind, CapabilityTurnKind.COMPLETED)
        self.assertIsInstance(action, ActionProposalCapability)
        self.assertIsInstance(mutation, GraphMutationProposalCapability)
        self.assertEqual(
            tuple(request.cognitive_capability_id for request in backend.requests),
            ("action_proposal", "graph_mutation_proposal"),
        )

    async def test_planner_filters_tools_to_runtime_available_ids(self) -> None:
        backend = CapabilityResponseBackend(
            {
                "task_graph_proposal": {
                    "nodes": [
                        {
                            "node_key": "research",
                            "goal": "Research",
                            "expected_output": "Evidence",
                            "requested_strategy_id": "research",
                        }
                    ]
                }
            }
        )
        settings = CapabilityInferenceSettings(
            tool_intent=ToolIntentMode.ALLOWED,
            eligible_tools=(
                ToolSpecification(
                    capability_id="search",
                    name="Search",
                    description="Search",
                ),
                ToolSpecification(
                    capability_id="filesystem",
                    name="Filesystem",
                    description="Read files",
                ),
            ),
        )
        capability = GatewayTaskGraphProposalCapability(
            gateway=gateway_with(backend),
            draft_validator=CapabilityDraftValidator(),
            settings=settings,
        )

        await capability.propose(
            TaskPlanningRequest(
                task="Research",
                available_strategies=("research",),
                available_execution_capability_ids=("search",),
            )
        )

        self.assertEqual(
            tuple(tool.capability_id for tool in backend.requests[0].eligible_tools),
            ("search",),
        )

    async def test_required_tool_without_runtime_eligibility_is_rejected(self) -> None:
        backend = CapabilityResponseBackend({})
        capability = GatewayTaskGraphProposalCapability(
            gateway=gateway_with(backend),
            draft_validator=CapabilityDraftValidator(),
            settings=CapabilityInferenceSettings(
                tool_intent=ToolIntentMode.REQUIRED,
                eligible_tools=(
                    ToolSpecification(
                        capability_id="search",
                        name="Search",
                        description="Search",
                    ),
                ),
            ),
        )

        with self.assertRaisesRegex(
            CapabilityExecutionPolicyError,
            "no Runtime-eligible tools",
        ):
            await capability.propose(
                TaskPlanningRequest(
                    task="Research",
                    available_strategies=("research",),
                )
            )
        self.assertEqual(backend.requests, [])

    async def test_semantic_model_validation_rejects_cyclic_graph(self) -> None:
        backend = CapabilityResponseBackend(
            {
                "task_graph_proposal": {
                    "nodes": [
                        {
                            "node_key": "a",
                            "goal": "A",
                            "dependency_keys": ["b"],
                            "expected_output": "A",
                            "requested_strategy_id": "research",
                        },
                        {
                            "node_key": "b",
                            "goal": "B",
                            "dependency_keys": ["a"],
                            "expected_output": "B",
                            "requested_strategy_id": "research",
                        },
                    ]
                }
            }
        )
        capability = GatewayTaskGraphProposalCapability(
            gateway=gateway_with(backend),
            draft_validator=CapabilityDraftValidator(),
        )

        with self.assertRaisesRegex(
            ResponseSchemaValidationError,
            "must be acyclic",
        ):
            await capability.propose(
                TaskPlanningRequest(
                    task="Research",
                    available_strategies=("research",),
                )
            )

    async def test_cross_request_validation_runs_after_schema_validation(self) -> None:
        backend = CapabilityResponseBackend(
            {
                "task_graph_proposal": {
                    "nodes": [
                        {
                            "node_key": "research",
                            "goal": "Research",
                            "expected_output": "Evidence",
                            "requested_strategy_id": "unknown",
                        }
                    ]
                }
            }
        )
        capability = GatewayTaskGraphProposalCapability(
            gateway=gateway_with(backend),
            draft_validator=CapabilityDraftValidator(),
        )

        with self.assertRaisesRegex(
            CapabilityResultValidationError,
            "unavailable strategies",
        ):
            await capability.propose(
                TaskPlanningRequest(
                    task="Research",
                    available_strategies=("research",),
                )
            )


if __name__ == "__main__":
    unittest.main()
