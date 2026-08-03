"""Failure analysis and proposal-only optimization tests."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID

from pydantic import ValidationError

from adaptive_agent_runtime.evaluation import (
    ConservativeOptimizationAgent,
    ConservativeOptimizationPolicy,
    DeterministicFailureAnalyzer,
    EvaluationComponent,
    EvaluationFinding,
    EvaluationResult,
    EvaluationScope,
    EvaluationVerdict,
    FailureAnalysis,
    FindingSeverity,
    ProposalStatus,
)


_NOW = datetime(2026, 8, 2, 2, 0, tzinfo=timezone.utc)


def _tool_failure(run_number: int, *, confidence: float = 1.0) -> EvaluationResult:
    run_id = UUID(int=2000 + run_number)
    trace_id = UUID(int=3000 + run_number)
    finding = EvaluationFinding(
        finding_id=UUID(int=4000 + run_number),
        code="component.tool.capability_mismatch",
        severity=FindingSeverity.ERROR,
        component=EvaluationComponent.TOOL,
        summary="Tool selection did not satisfy the Capability requirement.",
        assessment_confidence=confidence,
        evidence_fact_ids=(UUID(int=5000 + run_number),),
    )
    return EvaluationResult(
        evaluation_id=UUID(int=6000 + run_number),
        trace_id=trace_id,
        run_id=run_id,
        task_id=UUID(int=7000 + run_number),
        evaluator_id="evaluation.component.tool",
        evaluator_version="1",
        scope=EvaluationScope.COMPONENT,
        component=EvaluationComponent.TOOL,
        verdict=EvaluationVerdict.FAIL,
        score=0.7,
        assessment_confidence=confidence,
        findings=(finding,),
        evaluated_at=_NOW + timedelta(minutes=run_number),
    )


class FailureAnalysisTests(unittest.TestCase):
    def test_candidate_cannot_claim_confidence_or_runs_outside_pattern(self) -> None:
        analysis = DeterministicFailureAnalyzer().analyze(
            (_tool_failure(1), _tool_failure(2))
        )
        pattern = analysis.patterns[0].model_copy(
            update={"pattern_confidence": 0.2}
        )
        candidate = analysis.candidates[0]

        with self.assertRaises(ValidationError):
            FailureAnalysis(
                patterns=(pattern,),
                candidates=(candidate,),
                analyzed_at=analysis.analyzed_at,
            )

        forged_runs = candidate.model_copy(
            update={"affected_run_ids": (*candidate.affected_run_ids, UUID(int=9999))}
        )
        with self.assertRaises(ValidationError):
            FailureAnalysis(
                patterns=analysis.patterns,
                candidates=(forged_runs,),
                analyzed_at=analysis.analyzed_at,
            )

    def test_repeated_failure_becomes_pattern_root_cause_and_candidate(self) -> None:
        first = _tool_failure(1)
        second = _tool_failure(2)
        analyzer = DeterministicFailureAnalyzer()

        analysis = analyzer.analyze((second, first))
        reordered = analyzer.analyze((first, second))

        self.assertEqual(analysis, reordered)
        self.assertEqual(len(analysis.patterns), 1)
        self.assertEqual(len(analysis.candidates), 1)
        pattern = analysis.patterns[0]
        candidate = analysis.candidates[0]
        self.assertEqual(pattern.root_cause.code, "tool.capability_mismatch")
        self.assertEqual(pattern.affected_run_ids, tuple(sorted(
            (first.run_id, second.run_id),
            key=str,
        )))
        self.assertEqual(candidate.pattern_id, pattern.pattern_id)
        self.assertEqual(candidate.target_component, EvaluationComponent.TOOL)
        self.assertEqual(
            candidate.change_kind,
            "tool.selection_policy.review",
        )
        self.assertEqual(len(candidate.evidence_finding_ids), 2)

    def test_non_failure_findings_do_not_create_patterns(self) -> None:
        source = _tool_failure(1)
        warning = source.findings[0].model_copy(
            update={"severity": FindingSeverity.WARNING}
        )
        result = source.model_copy(update={"findings": (warning,)})

        analysis = DeterministicFailureAnalyzer().analyze((result,))

        self.assertEqual(analysis.patterns, ())
        self.assertEqual(analysis.candidates, ())


class OptimizationProposalTests(unittest.TestCase):
    def test_new_cross_run_evidence_produces_a_new_proposal_snapshot(self) -> None:
        analyzer = DeterministicFailureAnalyzer()
        agent = ConservativeOptimizationAgent()
        policy = ConservativeOptimizationPolicy()
        two_runs = analyzer.analyze((_tool_failure(1), _tool_failure(2)))
        three_runs = analyzer.analyze(
            (_tool_failure(1), _tool_failure(2), _tool_failure(3))
        )

        first = agent.propose(two_runs, policy)[0]
        expanded = agent.propose(three_runs, policy)[0]

        self.assertNotEqual(first.proposal_id, expanded.proposal_id)
        self.assertNotEqual(
            first.source_candidate_ids,
            expanded.source_candidate_ids,
        )

    def test_recurrent_high_confidence_candidate_yields_proposal_only(self) -> None:
        analysis = DeterministicFailureAnalyzer().analyze(
            (_tool_failure(1), _tool_failure(2))
        )
        agent = ConservativeOptimizationAgent()
        policy = ConservativeOptimizationPolicy()

        proposals = agent.propose(analysis, policy)

        self.assertEqual(proposals, agent.propose(analysis, policy))
        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(proposal.status, ProposalStatus.PROPOSED)
        self.assertTrue(proposal.change_spec["proposal_only"])
        self.assertTrue(proposal.change_spec["config_patch"])
        self.assertEqual(proposal.target_component, EvaluationComponent.TOOL)
        self.assertFalse(hasattr(agent, "apply"))
        self.assertFalse(hasattr(agent, "execute"))
        self.assertNotIn("approval", type(proposal).model_fields)
        self.assertNotIn("applied", type(proposal).model_fields)

    def test_single_run_and_low_confidence_candidates_are_filtered(self) -> None:
        agent = ConservativeOptimizationAgent()
        policy = ConservativeOptimizationPolicy()
        single_run = DeterministicFailureAnalyzer().analyze((_tool_failure(1),))
        low_confidence = DeterministicFailureAnalyzer().analyze(
            (_tool_failure(1, confidence=0.5), _tool_failure(2, confidence=0.5))
        )
        same_run_second = _tool_failure(2).model_copy(
            update={"run_id": _tool_failure(1).run_id}
        )
        repeated_in_one_run = DeterministicFailureAnalyzer().analyze(
            (_tool_failure(1), same_run_second)
        )

        self.assertEqual(agent.propose(single_run, policy), ())
        self.assertEqual(agent.propose(low_confidence, policy), ())
        self.assertEqual(agent.propose(repeated_in_one_run, policy), ())


if __name__ == "__main__":
    unittest.main()
