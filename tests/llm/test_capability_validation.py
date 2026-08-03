from __future__ import annotations

import unittest

from adaptive_agent_runtime.llm import (
    ActionProposalDraft,
    ActionProposalRequest,
    CapabilityDraftValidator,
    CapabilityResultValidationError,
    CompressedContextDraft,
    CompressionRequest,
    EvidenceReference,
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
    MemoryCandidateDraft,
    MemoryConditionDraft,
    MemoryEvolutionDraft,
    MemoryExtractionRequest,
    PlanningActionCandidate,
    ReasoningContext,
    ReasoningResult,
    TaskGraphDraft,
    TaskNodeDraft,
    TaskPlanningRequest,
)


def evidence(reference_id: str) -> EvidenceReference:
    return EvidenceReference(
        reference_id=reference_id,
        kind="test",
    )


class CapabilityDraftValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = CapabilityDraftValidator()

    def test_reasoning_evidence_must_come_from_request(self) -> None:
        request = ReasoningContext(
            goal="Analyze",
            evidence=(evidence("evidence:1"),),
        )
        valid = ReasoningResult(
            conclusions=("Supported",),
            evidence_reference_ids=("evidence:1",),
        )
        invalid = ReasoningResult(
            conclusions=("Unsupported",),
            evidence_reference_ids=("evidence:missing",),
        )

        self.assertIs(self.validator.validate_reasoning(request, valid), valid)
        with self.assertRaisesRegex(
            CapabilityResultValidationError,
            "unknown evidence references",
        ):
            self.validator.validate_reasoning(request, invalid)

    def test_task_graph_strategies_must_be_runtime_supplied(self) -> None:
        request = TaskPlanningRequest(
            task="Research",
            available_strategies=("research",),
        )
        graph = TaskGraphDraft(
            nodes=(
                TaskNodeDraft(
                    node_key="report",
                    goal="Report",
                    expected_output="Markdown",
                    requested_strategy_id="unavailable",
                ),
            )
        )

        with self.assertRaisesRegex(
            CapabilityResultValidationError,
            "unavailable strategies",
        ):
            self.validator.validate_task_graph(request, graph)

    def test_action_proposal_stays_inside_runtime_ready_set(self) -> None:
        request = ActionProposalRequest(
            task="Research",
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
        invalid = ActionProposalDraft(
            node_key="unready",
            rationale="Skip Runtime scheduling",
            evidence_reference_ids=("runtime:missing",),
        )

        with self.assertRaises(CapabilityResultValidationError) as captured:
            self.validator.validate_action_proposal(request, invalid)
        self.assertEqual(len(captured.exception.violations), 2)

    def test_graph_mutations_are_symbolic_allowlisted_and_evidence_bound(
        self,
    ) -> None:
        request = GraphMutationProposalRequest(
            task="Research",
            trigger_node_key="company",
            trigger_observation={"company": "Tesla"},
            existing_node_keys=("company", "risk"),
            allowed_new_node_keys=("news",),
            available_strategies=("research",),
            evidence=(evidence("observation:company"),),
        )
        proposal = GraphMutationProposalDraft(
            operations=(
                GraphMutationOperationDraft(
                    kind=GraphMutationOperationKind.ADD_NODE,
                    node=TaskNodeDraft(
                        node_key="unapproved",
                        goal="Unapproved",
                        dependency_keys=("missing",),
                        expected_output="Output",
                        requested_strategy_id="shell",
                    ),
                    reason="Escape the graph boundary",
                    evidence_reference_ids=("observation:missing",),
                ),
                GraphMutationOperationDraft(
                    kind=GraphMutationOperationKind.ADD_DEPENDENCY,
                    node_key="risk",
                    dependency_key="unapproved",
                    reason="Reference an unapproved node",
                    evidence_reference_ids=("observation:company",),
                ),
            ),
            rationale="Invalid batch",
        )

        with self.assertRaises(CapabilityResultValidationError) as captured:
            self.validator.validate_graph_mutation_proposal(request, proposal)
        self.assertGreaterEqual(len(captured.exception.violations), 4)

    def test_generated_artifact_preserves_media_type_and_evidence(self) -> None:
        request = GenerationRequest(
            instruction="Write",
            media_type="text/markdown",
            evidence=(evidence("evidence:1"),),
        )
        artifact = GeneratedArtifactDraft(
            media_type="text/plain",
            content="Report",
            evidence_reference_ids=("evidence:missing",),
        )

        with self.assertRaises(CapabilityResultValidationError) as captured:
            self.validator.validate_generation(request, artifact)
        self.assertEqual(len(captured.exception.violations), 2)

    def test_judge_findings_cannot_invent_evidence(self) -> None:
        request = JudgeRequest(
            subject={"status": "complete"},
            criteria=("grounded",),
            evidence_catalog=(evidence("trace:1"),),
        )
        assessment = JudgeAssessmentDraft(
            summary="Unsupported finding",
            findings=(
                JudgeFindingDraft(
                    code="quality.unsupported",
                    severity=JudgeSeverity.ERROR,
                    summary="Unsupported",
                    assessment_confidence=0.8,
                    evidence_reference_ids=("trace:missing",),
                ),
            ),
        )

        with self.assertRaisesRegex(
            CapabilityResultValidationError,
            "trace:missing",
        ):
            self.validator.validate_judge(request, assessment)

    def test_compression_must_meet_budget_and_source_boundary(self) -> None:
        request = CompressionRequest(
            source_reference_id="context:1",
            content="Long context",
            original_estimated_tokens=100,
            target_max_tokens=40,
        )
        compressed = CompressedContextDraft(
            content="Still too long",
            core_conclusions=("Conclusion",),
            source_reference_ids=("context:other",),
            estimated_tokens=50,
        )

        with self.assertRaises(CapabilityResultValidationError) as captured:
            self.validator.validate_compression(request, compressed)
        self.assertEqual(len(captured.exception.violations), 2)

    def test_memory_candidates_are_bound_to_catalogs(self) -> None:
        request = MemoryExtractionRequest(
            observations={"fact": "value"},
            evidence_catalog=(evidence("observation:1"),),
            existing_memory_reference_ids=("memory:known",),
        )
        candidate = MemoryCandidateDraft(
            memory_key="preference",
            content={"format": "markdown"},
            condition=MemoryConditionDraft(),
            evidence_reference_ids=("observation:missing",),
            confidence=0.9,
            evolution=MemoryEvolutionDraft.MODIFY,
            target_memory_reference="memory:missing",
        )

        with self.assertRaises(CapabilityResultValidationError) as captured:
            self.validator.validate_memory_extraction(request, (candidate,))
        self.assertEqual(len(captured.exception.violations), 2)


if __name__ == "__main__":
    unittest.main()
