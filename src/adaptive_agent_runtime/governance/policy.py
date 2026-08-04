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
        ),
    )
