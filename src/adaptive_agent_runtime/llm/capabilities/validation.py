"""Cross-request validation for untrusted cognitive capability drafts."""

from __future__ import annotations

from adaptive_agent_runtime.llm.capabilities.models import (
    ActionProposalDraft,
    ActionProposalRequest,
    CompressedContextDraft,
    CompressionRequest,
    EvidenceReference,
    GeneratedArtifactDraft,
    GenerationRequest,
    GraphMutationOperationKind,
    GraphMutationProposalDraft,
    GraphMutationProposalRequest,
    JudgeAssessmentDraft,
    JudgeRequest,
    MemoryCandidateDraft,
    MemoryExtractionRequest,
    ReasoningContext,
    ReasoningResult,
    TaskGraphDraft,
    TaskPlanningRequest,
)
from adaptive_agent_runtime.llm.errors import CapabilityResultValidationError


class CapabilityDraftValidator:
    """Validate a completed draft against the exact request that produced it."""

    module_id = "llm.capability_draft_validator"

    def validate_reasoning(
        self,
        request: ReasoningContext,
        result: ReasoningResult,
    ) -> ReasoningResult:
        violations = _unknown_references(
            result.evidence_reference_ids,
            _evidence_ids(request.evidence),
            "reasoning result",
        )
        self._raise_if_invalid("reasoning", violations)
        return result

    def validate_task_graph(
        self,
        request: TaskPlanningRequest,
        result: TaskGraphDraft,
    ) -> TaskGraphDraft:
        unknown = sorted(
            {
                node.requested_strategy_id
                for node in result.nodes
                if node.requested_strategy_id not in request.available_strategies
            }
        )
        violations = (
            (
                "task graph requested unavailable strategies: "
                + ", ".join(unknown)
            ),
        ) if unknown else ()
        self._raise_if_invalid("task_graph_proposal", violations)
        return result

    def validate_action_proposal(
        self,
        request: ActionProposalRequest,
        result: ActionProposalDraft,
    ) -> ActionProposalDraft:
        candidates = {candidate.node_key for candidate in request.candidates}
        violations: list[str] = []
        if result.node_key not in candidates:
            violations.append(
                "action proposal selected a node outside the Runtime ready set"
            )
        violations.extend(
            _unknown_references(
                result.evidence_reference_ids,
                _evidence_ids(request.evidence),
                "action proposal",
            )
        )
        self._raise_if_invalid("action_proposal", tuple(violations))
        return result

    def validate_graph_mutation_proposal(
        self,
        request: GraphMutationProposalRequest,
        result: GraphMutationProposalDraft,
    ) -> GraphMutationProposalDraft:
        violations: list[str] = []
        existing = set(request.existing_node_keys)
        allowed_new = set(request.allowed_new_node_keys)
        added_nodes = tuple(
            operation.node
            for operation in result.operations
            if operation.kind is GraphMutationOperationKind.ADD_NODE
            and operation.node is not None
        )
        added_keys = tuple(node.node_key for node in added_nodes)
        if len(set(added_keys)) != len(added_keys):
            violations.append("graph mutation proposal adds duplicate node keys")
        unknown_new = sorted(set(added_keys).difference(allowed_new))
        if unknown_new:
            violations.append(
                "graph mutation proposal adds nodes outside the Runtime allowlist: "
                + ", ".join(unknown_new)
            )
        known_after = existing.union(added_keys)
        available_strategies = set(request.available_strategies)
        for node in added_nodes:
            if node.requested_strategy_id not in available_strategies:
                violations.append(
                    "graph mutation proposal requested an unavailable strategy: "
                    + node.requested_strategy_id
                )
            unknown_dependencies = sorted(
                set(node.dependency_keys).difference(known_after)
            )
            if unknown_dependencies:
                violations.append(
                    f"new node '{node.node_key}' references unknown dependencies: "
                    + ", ".join(unknown_dependencies)
                )
        signatures: list[tuple[str, str, str | None]] = []
        for operation in result.operations:
            violations.extend(
                _unknown_references(
                    operation.evidence_reference_ids,
                    _evidence_ids(request.evidence),
                    "graph mutation proposal",
                )
            )
            if operation.kind is GraphMutationOperationKind.ADD_NODE:
                assert operation.node is not None
                signatures.append((operation.kind.value, operation.node.node_key, None))
                continue
            assert operation.node_key is not None
            assert operation.dependency_key is not None
            signatures.append(
                (
                    operation.kind.value,
                    operation.node_key,
                    operation.dependency_key,
                )
            )
            unknown = sorted(
                {operation.node_key, operation.dependency_key}.difference(
                    known_after
                )
            )
            if unknown:
                violations.append(
                    "graph dependency proposal references unknown nodes: "
                    + ", ".join(unknown)
                )
        if len(set(signatures)) != len(signatures):
            violations.append("graph mutation proposal contains duplicates")
        self._raise_if_invalid("graph_mutation_proposal", tuple(violations))
        return result

    def validate_generation(
        self,
        request: GenerationRequest,
        result: GeneratedArtifactDraft,
    ) -> GeneratedArtifactDraft:
        violations: list[str] = []
        if result.media_type != request.media_type:
            violations.append(
                "generated artifact media type does not match the request"
            )
        violations.extend(
            _unknown_references(
                result.evidence_reference_ids,
                _evidence_ids(request.evidence),
                "generated artifact",
            )
        )
        self._raise_if_invalid("artifact_generation", tuple(violations))
        return result

    def validate_judge(
        self,
        request: JudgeRequest,
        result: JudgeAssessmentDraft,
    ) -> JudgeAssessmentDraft:
        references = tuple(
            reference_id
            for finding in result.findings
            for reference_id in finding.evidence_reference_ids
        )
        violations = _unknown_references(
            references,
            _evidence_ids(request.evidence_catalog),
            "judge assessment",
        )
        self._raise_if_invalid("evidence_judge", violations)
        return result

    def validate_compression(
        self,
        request: CompressionRequest,
        result: CompressedContextDraft,
    ) -> CompressedContextDraft:
        violations: list[str] = []
        if set(result.source_reference_ids) != {request.source_reference_id}:
            violations.append(
                "compressed context must reference only its source context"
            )
        if result.estimated_tokens > request.target_max_tokens:
            violations.append("compressed context exceeds its target token budget")
        self._raise_if_invalid("semantic_compression", tuple(violations))
        return result

    def validate_memory_extraction(
        self,
        request: MemoryExtractionRequest,
        results: tuple[MemoryCandidateDraft, ...],
    ) -> tuple[MemoryCandidateDraft, ...]:
        violations: list[str] = []
        allowed_evidence = _evidence_ids(request.evidence_catalog)
        allowed_memories = set(request.existing_memory_reference_ids)
        for index, result in enumerate(results):
            violations.extend(
                _unknown_references(
                    result.evidence_reference_ids,
                    allowed_evidence,
                    f"memory candidate {index}",
                )
            )
            target = result.target_memory_reference
            if target is not None and target not in allowed_memories:
                violations.append(
                    f"memory candidate {index} targets unknown memory: {target}"
                )
        self._raise_if_invalid("memory_extraction", tuple(violations))
        return results

    @staticmethod
    def _raise_if_invalid(
        capability_id: str,
        violations: tuple[str, ...],
    ) -> None:
        if violations:
            raise CapabilityResultValidationError(capability_id, violations)


def _evidence_ids(evidence: tuple[EvidenceReference, ...]) -> set[str]:
    return {item.reference_id for item in evidence}


def _unknown_references(
    references: tuple[str, ...],
    allowed: set[str],
    subject: str,
) -> tuple[str, ...]:
    unknown = sorted(set(references).difference(allowed))
    if not unknown:
        return ()
    return (
        f"{subject} cites unknown evidence references: " + ", ".join(unknown),
    )
