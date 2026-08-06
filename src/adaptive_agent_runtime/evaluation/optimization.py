"""Legacy/test-only Phase 5 proposal demo.

This compatibility implementation is not part of Research default composition
and is not the governed Phase 4-A Optimization Proposal lifecycle.
"""

from __future__ import annotations

from pydantic import JsonValue

from adaptive_agent_runtime.evaluation.models import (
    ConservativeOptimizationPolicy,
    FailureAnalysis,
    OptimizationProposal,
    stable_evaluation_id,
)


class ConservativeOptimizationAgent:
    """Legacy deterministic test fixture; never a default Runtime decision."""
    module_id = "evaluation.optimization_agent.conservative"

    def propose(
        self,
        analysis: FailureAnalysis,
        policy: ConservativeOptimizationPolicy,
    ) -> tuple[OptimizationProposal, ...]:
        patterns = {item.pattern_id: item for item in analysis.patterns}
        proposals: list[OptimizationProposal] = []
        for candidate in analysis.candidates:
            pattern = patterns[candidate.pattern_id]
            if len(candidate.affected_run_ids) < policy.min_affected_runs:
                continue
            if pattern.pattern_confidence < policy.min_pattern_confidence:
                continue
            if candidate.assessment_confidence < policy.min_pattern_confidence:
                continue
            if candidate.expected_benefit_score < policy.min_expected_benefit:
                continue
            if len(candidate.evidence_finding_ids) < policy.min_evidence_findings:
                continue
            proposal_id = stable_evaluation_id(
                "optimization-proposal",
                candidate.candidate_id,
                pattern.pattern_id,
            )
            proposals.append(
                OptimizationProposal(
                    proposal_id=proposal_id,
                    source_pattern_ids=(pattern.pattern_id,),
                    source_candidate_ids=(candidate.candidate_id,),
                    target_component=candidate.target_component,
                    change_kind=candidate.change_kind,
                    change_spec={
                        "proposal_only": True,
                        "pattern_key": pattern.pattern_key,
                        "candidate_id": str(candidate.candidate_id),
                        "recommended_change_kind": candidate.change_kind,
                        "config_patch": self._config_patch(
                            candidate.change_kind
                        ),
                    },
                    rationale=candidate.rationale,
                    expected_benefit=candidate.expected_benefit,
                    proposal_confidence=min(
                        pattern.pattern_confidence,
                        candidate.assessment_confidence,
                    ),
                    validation_plan=(
                        "Evaluate the proposed change in an isolated replay.",
                        "Compare outcome and trajectory scores against the baseline.",
                    ),
                    rollback_plan=(
                        "Retain the current configuration as the rollback baseline.",
                    ),
                    created_at=pattern.last_seen,
                )
            )
        return tuple(sorted(proposals, key=lambda item: str(item.proposal_id)))

    @staticmethod
    def _config_patch(change_kind: str) -> dict[str, JsonValue]:
        patches: dict[str, dict[str, JsonValue]] = {
            "tool.selection_policy.review": {
                "capability_prevalidation": True,
            },
            "tool.execution_policy.review": {
                "max_retries": 2,
            },
            "orchestration.strategy.review": {
                "failure_replanning_enabled": True,
            },
            "context_memory.policy.review": {
                "compression_trigger_ratio": 0.75,
            },
            "runtime.configuration.review": {
                "strict_runtime_validation": True,
            },
        }
        return patches.get(change_kind, {"review_required": True})
