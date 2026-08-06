"""Static Governance policy and deterministic rule matching."""

from __future__ import annotations

from adaptive_agent_runtime.governance.models import (
    GovernancePolicy,
    GovernanceRequest,
    GovernanceRule,
    GovernanceScope,
    RiskLevel,
    RuleEffect,
    RuleEvaluation,
)


class DeterministicRuleEvaluator:
    module_id = "governance.rule_evaluator.deterministic"

    def evaluate(
        self,
        request: GovernanceRequest,
        policy: GovernancePolicy,
    ) -> RuleEvaluation:
        matching = tuple(
            sorted(
                (
                    rule
                    for rule in policy.rules
                    if rule.enabled and self._matches(rule, request)
                ),
                key=lambda rule: (
                    -rule.priority,
                    0 if rule.effect is RuleEffect.DENY else 1,
                    rule.rule_id,
                ),
            )
        )
        if not matching:
            return RuleEvaluation(reason="No Governance rule matched the request.")
        winner = matching[0]
        return RuleEvaluation(
            matched_rule_ids=(winner.rule_id,),
            effect=winner.effect,
            reason=f"Governance rule '{winner.rule_id}' matched.",
        )

    @staticmethod
    def _matches(rule: GovernanceRule, request: GovernanceRequest) -> bool:
        if rule.scopes and request.scope not in rule.scopes:
            return False
        if (
            rule.operations
            and "*" not in rule.operations
            and request.operation not in rule.operations
        ):
            return False
        if rule.risk_levels and request.risk not in rule.risk_levels:
            return False
        return True


def default_governance_policy() -> GovernancePolicy:
    """Small demonstration policy; applications may inject another snapshot."""

    return GovernancePolicy(
        policy_id="runtime.default",
        version="1",
        rules=(
            GovernanceRule(
                rule_id="allow.low_risk.tool_call",
                description="Allow ordinary low-risk Tool calls.",
                effect=RuleEffect.ALLOW,
                scopes=(GovernanceScope.ACTION,),
                operations=("tool.call",),
                risk_levels=(RiskLevel.LOW,),
                priority=100,
            ),
            GovernanceRule(
                rule_id="allow.low_risk.tool_selection",
                description=(
                    "Allow reversible binding to a Runtime-filtered Tool Provider."
                ),
                effect=RuleEffect.ALLOW,
                scopes=(GovernanceScope.ACTION,),
                operations=("tool.selection.bind",),
                risk_levels=(RiskLevel.LOW,),
                priority=100,
            ),
            GovernanceRule(
                rule_id="allow.low_risk.node_select",
                description=(
                    "Allow selection from a Runtime-validated ready-node set."
                ),
                effect=RuleEffect.ALLOW,
                scopes=(GovernanceScope.ACTION,),
                operations=("node.select",),
                risk_levels=(RiskLevel.LOW,),
                priority=100,
            ),
            GovernanceRule(
                rule_id="allow.low_risk.context_cleanup",
                description="Allow temporary Context cleanup.",
                effect=RuleEffect.ALLOW,
                scopes=(GovernanceScope.STATE,),
                operations=("context.compress", "context.restore"),
                risk_levels=(RiskLevel.LOW,),
                priority=100,
            ),
            GovernanceRule(
                rule_id="allow.low_risk.root_cause_record",
                description=(
                    "Allow append-only advisory Root Cause assessments that do "
                    "not alter execution or deterministic Evaluation results."
                ),
                effect=RuleEffect.ALLOW,
                scopes=(GovernanceScope.STATE,),
                operations=("evaluation.root_cause.record",),
                risk_levels=(RiskLevel.LOW,),
                priority=100,
            ),
            GovernanceRule(
                rule_id="allow.low_risk.report_commit",
                description="Allow one reversible, provenance-bound report commit.",
                effect=RuleEffect.ALLOW,
                scopes=(GovernanceScope.STATE,),
                operations=("workspace.report.commit",),
                risk_levels=(RiskLevel.LOW,),
                priority=100,
            ),
            GovernanceRule(
                rule_id="allow.low_risk.memory_recall_commit",
                description=(
                    "Allow an immutable, scope-filtered Memory Recall Bundle commit."
                ),
                effect=RuleEffect.ALLOW,
                scopes=(GovernanceScope.STATE,),
                operations=("memory.recall.commit",),
                risk_levels=(RiskLevel.LOW,),
                priority=100,
            ),
            GovernanceRule(
                rule_id="allow.low_risk.experience_metadata_commit",
                description=(
                    "Allow append-only, Runtime-evidenced Experience Metadata."
                ),
                effect=RuleEffect.ALLOW,
                scopes=(GovernanceScope.STATE,),
                operations=("experience.metadata.commit",),
                risk_levels=(RiskLevel.LOW,),
                priority=100,
            ),
            GovernanceRule(
                rule_id="allow.low_risk.decision_feedback_commit",
                description=(
                    "Allow append-only, non-behavioral Decision outcome feedback."
                ),
                effect=RuleEffect.ALLOW,
                scopes=(GovernanceScope.STATE,),
                operations=("decision.feedback.commit",),
                risk_levels=(RiskLevel.LOW,),
                priority=100,
            ),
            GovernanceRule(
                rule_id="allow.low_risk.learning_insight_commit",
                description=(
                    "Allow append-only, non-behavioral cross-run Learning Insights."
                ),
                effect=RuleEffect.ALLOW,
                scopes=(GovernanceScope.STATE,),
                operations=("learning.insight.commit",),
                risk_levels=(RiskLevel.LOW,),
                priority=100,
            ),
            GovernanceRule(
                rule_id="allow.low_risk.optimization_proposal_commit",
                description=(
                    "Allow append-only Optimization Proposal storage without "
                    "configuration activation or Runtime behavior change."
                ),
                effect=RuleEffect.ALLOW,
                scopes=(GovernanceScope.EVOLUTION,),
                operations=("optimization.proposal.commit",),
                risk_levels=(RiskLevel.LOW,),
                priority=100,
            ),
            GovernanceRule(
                rule_id="allow.bounded.optimization_apply_commit",
                description=(
                    "Allow an explicit, reversible planner.max_nodes activation "
                    "whose final Effect is bound to the active baseline."
                ),
                effect=RuleEffect.ALLOW,
                scopes=(GovernanceScope.EVOLUTION,),
                operations=("optimization.apply.commit",),
                risk_levels=(RiskLevel.MEDIUM,),
                priority=100,
            ),
            GovernanceRule(
                rule_id="allow.explicit.optimization_rollback_commit",
                description=(
                    "Allow explicit restoration of a prior immutable Runtime "
                    "configuration snapshot as a new revision."
                ),
                effect=RuleEffect.ALLOW,
                scopes=(GovernanceScope.EVOLUTION,),
                operations=("optimization.rollback.commit",),
                risk_levels=(RiskLevel.LOW,),
                priority=100,
            ),
        ),
    )
