from __future__ import annotations

import unittest
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from adaptive_agent_runtime.llm import (
    AgentTargetProfile,
    BackendExecutionMode,
    BackendKind,
    BackendDelegatedAccess,
    CapabilityTurnKind,
    CapabilityTurnResult,
    CompressionRequest,
    EvidenceReference,
    GraphMutationOperationDraft,
    GraphMutationProposalDraft,
    InferenceRequest,
    InferenceRequirements,
    InferenceTargetProfile,
    InferenceUsage,
    JudgeFindingDraft,
    JudgeSeverity,
    MemoryCandidateDraft,
    MemoryConditionDraft,
    MemoryEvolutionDraft,
    ModelResponseKind,
    NormalizedModelResponse,
    ReasoningResult,
    StructuredOutputLevel,
    TaskGraphDraft,
    TaskNodeDraft,
    ToolIntentDraft,
    ToolIntentMode,
    ToolSpecification,
)


class InferenceModelTests(unittest.TestCase):
    def test_negotiated_cognitive_capabilities_are_explicit_and_unique(self) -> None:
        with self.assertRaisesRegex(ValidationError, "cannot be empty"):
            InferenceTargetProfile(
                target_id="invalid/empty-capabilities",
                backend_id="invalid",
                backend_kind=BackendKind.LOCAL,
                adapter_version="1",
                supported_cognitive_capability_ids=(),
            )
        with self.assertRaisesRegex(ValidationError, "must be unique"):
            InferenceTargetProfile(
                target_id="invalid/duplicate-capabilities",
                backend_id="invalid",
                backend_kind=BackendKind.LOCAL,
                adapter_version="1",
                supported_cognitive_capability_ids=("reasoning", "reasoning"),
            )

    def test_inference_request_is_deeply_immutable(self) -> None:
        source: dict[str, Any] = {"items": [{"value": 1}]}
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input=source,
        )

        source["items"][0]["value"] = 9

        frozen = cast(Any, request.input)
        self.assertEqual(frozen["items"][0]["value"], 1)
        with self.assertRaises(TypeError):
            frozen["items"][0]["value"] = 2

    def test_validated_copy_cannot_bypass_request_invariants(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="reasoning",
            input="goal",
        )

        with self.assertRaises(ValidationError):
            request.model_copy(
                update={
                    "requirements": InferenceRequirements(
                        tool_intent=ToolIntentMode.REQUIRED
                    )
                }
            )

    def test_llm_json_rejects_invalid_keys_and_non_finite_numbers(self) -> None:
        with self.assertRaises(ValidationError):
            InferenceRequest(
                cognitive_capability_id="reasoning",
                input=cast(Any, {1: "invalid"}),
            )
        with self.assertRaises(ValidationError):
            InferenceRequest(
                cognitive_capability_id="reasoning",
                input={"score": float("nan")},
            )

    def test_inference_profile_rejects_delegated_external_access(self) -> None:
        self.assertNotIn(
            "delegated_access",
            InferenceTargetProfile.model_fields,
        )
        with self.assertRaises(ValidationError):
            InferenceTargetProfile.model_validate(
                {
                    "target_id": "cli/inference",
                    "backend_id": "cli",
                    "backend_kind": BackendKind.CLI,
                    "adapter_version": "1",
                    "delegated_access": {"shell_execution": True},
                }
            )
        with self.assertRaises(ValidationError):
            InferenceTargetProfile.model_validate(
                {
                    "target_id": "cli/inference",
                    "backend_id": "cli",
                    "backend_kind": BackendKind.CLI,
                    "adapter_version": "1",
                    "execution_mode": BackendExecutionMode.AGENT,
                }
            )

        agent = AgentTargetProfile(
            target_id="cli/agent",
            backend_id="cli",
            backend_kind=BackendKind.CLI,
            adapter_version="1",
            delegated_access=BackendDelegatedAccess(
                filesystem_read=True,
                shell_execution=True,
            ),
        )
        self.assertEqual(agent.execution_mode, BackendExecutionMode.AGENT)
        self.assertTrue(agent.delegated_access.shell_execution)

    def test_json_schema_requirement_needs_schema(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "JSON Schema output requires a response_schema",
        ):
            InferenceRequest(
                cognitive_capability_id="judge",
                input={"subject": "result"},
                requirements=InferenceRequirements(
                    required_structured_output=StructuredOutputLevel.JSON_SCHEMA
                ),
            )

    def test_tool_intent_request_enforces_explicit_eligibility(self) -> None:
        tool = ToolSpecification(
            capability_id="weather",
            name="Weather",
            description="Read current weather.",
            input_schema={"type": "object"},
        )
        with self.assertRaisesRegex(
            ValidationError,
            "disabled tool intent cannot expose eligible tools",
        ):
            InferenceRequest(
                cognitive_capability_id="reasoning",
                input="weather",
                eligible_tools=(tool,),
            )
        with self.assertRaisesRegex(
            ValidationError,
            "required tool intent needs at least one eligible tool",
        ):
            InferenceRequest(
                cognitive_capability_id="reasoning",
                input="weather",
                requirements=InferenceRequirements(
                    tool_intent=ToolIntentMode.REQUIRED
                ),
            )

    def test_normalized_response_is_an_explicit_turn_state(self) -> None:
        request = InferenceRequest(
            cognitive_capability_id="generation",
            input="write",
        )
        null_output = NormalizedModelResponse(
            request_id=request.request_id,
            target_id="fake",
            kind=ModelResponseKind.OUTPUT,
            output=None,
        )
        self.assertIsNone(null_output.output)
        with self.assertRaisesRegex(
            ValidationError,
            "an output response requires an explicit output",
        ):
            NormalizedModelResponse(
                request_id=request.request_id,
                target_id="fake",
                kind=ModelResponseKind.OUTPUT,
            )
        with self.assertRaisesRegex(
            ValidationError,
            "an output response cannot contain tool intents",
        ):
            NormalizedModelResponse(
                request_id=request.request_id,
                target_id="fake",
                kind=ModelResponseKind.OUTPUT,
                output="done",
                tool_intents=(
                    ToolIntentDraft(
                        call_key="call-1",
                        capability_id="search",
                    ),
                ),
            )
        with self.assertRaisesRegex(
            ValidationError,
            "a tool-intent response requires tool intents",
        ):
            NormalizedModelResponse(
                request_id=request.request_id,
                target_id="fake",
                kind=ModelResponseKind.TOOL_INTENT,
            )

    def test_usage_cost_and_currency_are_atomic(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "monetary cost and currency must appear together",
        ):
            InferenceUsage(monetary_cost=0.1)

    def test_usage_total_matches_input_and_output_tokens(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "total tokens must equal input plus output tokens",
        ):
            InferenceUsage(
                input_tokens=2,
                output_tokens=3,
                total_tokens=99,
            )


class CognitiveDraftTests(unittest.TestCase):
    def test_capability_turn_returns_tool_intent_to_runtime(self) -> None:
        intent = ToolIntentDraft(
            call_key="search-1",
            capability_id="search",
            arguments={"query": "company news"},
        )
        turn = CapabilityTurnResult[ReasoningResult](
            kind=CapabilityTurnKind.TOOL_INTENT,
            tool_intents=(intent,),
        )

        self.assertIsNone(turn.result)
        self.assertEqual(turn.tool_intents, (intent,))
        with self.assertRaisesRegex(
            ValidationError,
            "a tool-intent turn requires tool intents",
        ):
            CapabilityTurnResult[ReasoningResult](
                kind=CapabilityTurnKind.TOOL_INTENT,
            )

    def test_task_graph_draft_uses_symbolic_keys_and_validates_dag(self) -> None:
        graph = TaskGraphDraft(
            nodes=(
                TaskNodeDraft(
                    node_key="research",
                    goal="Research the company",
                    expected_output="Evidence",
                    requested_strategy_id="research",
                ),
                TaskNodeDraft(
                    node_key="report",
                    goal="Write the report",
                    dependency_keys=("research",),
                    expected_output="Report",
                    requested_strategy_id="report",
                ),
            )
        )
        self.assertEqual(graph.nodes[1].dependency_keys, ("research",))
        self.assertNotIn("node_id", TaskNodeDraft.model_fields)

        with self.assertRaisesRegex(ValidationError, "unknown dependencies"):
            TaskGraphDraft(
                nodes=(
                    TaskNodeDraft(
                        node_key="report",
                        goal="Write",
                        dependency_keys=("missing",),
                        expected_output="Report",
                        requested_strategy_id="report",
                    ),
                )
            )

        with self.assertRaisesRegex(ValidationError, "must be acyclic"):
            TaskGraphDraft(
                nodes=(
                    TaskNodeDraft(
                        node_key="first",
                        goal="First",
                        dependency_keys=("second",),
                        expected_output="First output",
                        requested_strategy_id="research",
                    ),
                    TaskNodeDraft(
                        node_key="second",
                        goal="Second",
                        dependency_keys=("first",),
                        expected_output="Second output",
                        requested_strategy_id="research",
                    ),
                )
            )

    def test_authority_free_drafts_do_not_expose_runtime_identity(self) -> None:
        authority_fields: dict[type[BaseModel], tuple[str, ...]] = {
            TaskNodeDraft: ("node_id", "status", "observation"),
            TaskGraphDraft: ("graph_id", "version", "revision"),
            GraphMutationOperationDraft: (
                "mutation_id",
                "node_id",
                "dependency_id",
            ),
            GraphMutationProposalDraft: (
                "graph_id",
                "version",
                "apply",
            ),
            JudgeFindingDraft: ("finding_id", "evaluator_version", "created_at"),
            MemoryCandidateDraft: (
                "candidate_id",
                "memory_id",
                "revision",
                "created_at",
            ),
            CompressionRequest: ("context_id", "recovery_reference", "revision"),
        }
        for model, forbidden_fields in authority_fields.items():
            for field in forbidden_fields:
                with self.subTest(model=model.__name__, field=field):
                    self.assertNotIn(field, model.model_fields)

    def test_reasoning_result_rejects_hidden_reasoning_trace(self) -> None:
        with self.assertRaises(ValidationError):
            ReasoningResult.model_validate(
                {
                    "conclusions": ["Supported conclusion"],
                    "reasoning_trace": "private chain of thought",
                }
            )

    def test_judge_finding_requires_evidence(self) -> None:
        with self.assertRaises(ValidationError):
            JudgeFindingDraft(
                code="quality.missing",
                severity=JudgeSeverity.ERROR,
                summary="Required content is missing.",
                assessment_confidence=0.8,
                evidence_reference_ids=(),
            )

    def test_memory_draft_enforces_evolution_target_semantics(self) -> None:
        condition = MemoryConditionDraft(
            facts={"domain": "research"},
            required_tags=("research",),
        )
        extend = MemoryCandidateDraft(
            memory_key="research.preference",
            content={"format": "markdown"},
            condition=condition,
            evidence_reference_ids=("context:1",),
            confidence=0.9,
            evolution=MemoryEvolutionDraft.EXTEND,
        )
        self.assertIsNone(extend.target_memory_reference)

        with self.assertRaisesRegex(ValidationError, "require a target reference"):
            MemoryCandidateDraft(
                memory_key="research.preference",
                content={"format": "concise"},
                condition=condition,
                evidence_reference_ids=("context:2",),
                confidence=0.8,
                evolution=MemoryEvolutionDraft.MODIFY,
            )

    def test_evidence_reference_is_a_reference_not_runtime_evidence(self) -> None:
        reference = EvidenceReference(
            reference_id="trace:fact-1",
            kind="trace_fact",
            summary="The report omitted a required section.",
        )
        self.assertEqual(reference.reference_id, "trace:fact-1")
        self.assertNotIn("evidence_id", EvidenceReference.model_fields)


if __name__ == "__main__":
    unittest.main()
