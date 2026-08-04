"""Application-owned Research Task Graph definitions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

from adaptive_agent_runtime.llm import (
    GraphMutationOperationKind,
    GraphMutationProposalDraft,
    TaskGraphDraft,
)
from adaptive_agent_runtime.orchestration import (
    DynamicTaskGraph,
    GraphMutation,
    PlanningGraphEffect,
    TaskNode,
)


COMPANY_RESEARCH = "company_research"
FINANCIAL_DOCUMENT = "financial_document"
FINANCIAL_METRICS = "financial_metrics"
INDUSTRY_ANALYSIS = "industry_analysis"
COMPETITOR_ANALYSIS = "competitor_analysis"
NEWS_ANALYSIS = "news_analysis"
RISK_REVIEW = "risk_review"
REPORT_GENERATION = "report_generation"

RESEARCH_STRATEGY_ID = "research"
REVIEW_STRATEGY_ID = "review"
REPORT_STRATEGY_ID = "report"

RESEARCH_NODE_ROLES = (
    COMPANY_RESEARCH,
    FINANCIAL_DOCUMENT,
    FINANCIAL_METRICS,
    INDUSTRY_ANALYSIS,
    COMPETITOR_ANALYSIS,
    NEWS_ANALYSIS,
    RISK_REVIEW,
    REPORT_GENERATION,
)

_STRATEGY_BY_ROLE = {
    COMPANY_RESEARCH: RESEARCH_STRATEGY_ID,
    FINANCIAL_DOCUMENT: RESEARCH_STRATEGY_ID,
    FINANCIAL_METRICS: RESEARCH_STRATEGY_ID,
    INDUSTRY_ANALYSIS: RESEARCH_STRATEGY_ID,
    COMPETITOR_ANALYSIS: RESEARCH_STRATEGY_ID,
    NEWS_ANALYSIS: RESEARCH_STRATEGY_ID,
    RISK_REVIEW: REVIEW_STRATEGY_ID,
    REPORT_GENERATION: REPORT_STRATEGY_ID,
}

_REQUIRED_DEPENDENCIES = {
    FINANCIAL_DOCUMENT: {COMPANY_RESEARCH},
    FINANCIAL_METRICS: {FINANCIAL_DOCUMENT},
    INDUSTRY_ANALYSIS: {COMPANY_RESEARCH},
    COMPETITOR_ANALYSIS: {COMPANY_RESEARCH},
    NEWS_ANALYSIS: {COMPANY_RESEARCH},
    RISK_REVIEW: {
        FINANCIAL_METRICS,
        INDUSTRY_ANALYSIS,
        COMPETITOR_ANALYSIS,
        NEWS_ANALYSIS,
    },
    REPORT_GENERATION: {RISK_REVIEW},
}


@dataclass(frozen=True)
class ResearchTaskDefinition:
    """Initial graph plus the Application-owned node role map."""

    company: str
    initial_graph: DynamicTaskGraph
    nodes: Mapping[str, TaskNode]

    def node(self, role: str) -> TaskNode:
        return self.nodes[role]

    def discovery_mutations(self) -> tuple[GraphMutation, ...]:
        """Add news research after the company discovery node completes."""

        news = self.node(NEWS_ANALYSIS)
        if news.node_id in {node.node_id for node in self.initial_graph.nodes}:
            return ()
        risk = self.node(RISK_REVIEW)
        return (
            GraphMutation.add_node(
                news,
                reason="Company discovery identified a news-review requirement.",
            ),
            GraphMutation.add_dependency(
                risk.node_id,
                news.node_id,
                reason="Risk review must include the newly discovered news evidence.",
            ),
        )


def build_research_task(company: str) -> ResearchTaskDefinition:
    """Create a provider-neutral financial research task graph."""

    root = TaskNode(
        goal=f"Establish the research profile for {company}",
        expected_output="Company identity, business profile, and research scope",
        strategy_id=RESEARCH_STRATEGY_ID,
    )
    financial_document = TaskNode(
        goal=f"Analyze {company} financial statements",
        dependencies=(root.node_id,),
        expected_output="Normalized financial statement observations",
        strategy_id=RESEARCH_STRATEGY_ID,
    )
    financial_metrics = TaskNode(
        goal=f"Calculate investment metrics for {company}",
        dependencies=(financial_document.node_id,),
        expected_output="Margins, growth, leverage, and valuation indicators",
        strategy_id=RESEARCH_STRATEGY_ID,
    )
    industry = TaskNode(
        goal=f"Assess the industry position of {company}",
        dependencies=(root.node_id,),
        expected_output="Industry drivers, structure, and outlook",
        strategy_id=RESEARCH_STRATEGY_ID,
    )
    competitor = TaskNode(
        goal=f"Compare {company} with key competitors",
        dependencies=(root.node_id,),
        expected_output="Peer comparison and competitive differentiation",
        strategy_id=RESEARCH_STRATEGY_ID,
    )
    risk = TaskNode(
        goal=f"Independently review investment risks for {company}",
        dependencies=(
            financial_metrics.node_id,
            industry.node_id,
            competitor.node_id,
        ),
        expected_output="Independent risk findings and counterarguments",
        strategy_id=REVIEW_STRATEGY_ID,
    )
    report = TaskNode(
        goal=f"Generate the structured investment report for {company}",
        dependencies=(risk.node_id,),
        expected_output="A structured Markdown financial research report",
        strategy_id=REPORT_STRATEGY_ID,
    )
    news = TaskNode(
        goal=f"Review material news affecting {company}",
        dependencies=(root.node_id,),
        expected_output="Material news, catalysts, and controversies",
        strategy_id=RESEARCH_STRATEGY_ID,
    )
    nodes = MappingProxyType(
        {
            COMPANY_RESEARCH: root,
            FINANCIAL_DOCUMENT: financial_document,
            FINANCIAL_METRICS: financial_metrics,
            INDUSTRY_ANALYSIS: industry,
            COMPETITOR_ANALYSIS: competitor,
            NEWS_ANALYSIS: news,
            RISK_REVIEW: risk,
            REPORT_GENERATION: report,
        }
    )
    return ResearchTaskDefinition(
        company=company,
        initial_graph=DynamicTaskGraph(
            nodes=(
                root,
                financial_document,
                industry,
                competitor,
                financial_metrics,
                risk,
                report,
            )
        ),
        nodes=nodes,
    )


def build_research_task_from_draft(
    company: str,
    draft: TaskGraphDraft,
) -> ResearchTaskDefinition:
    """Validate an LLM graph proposal before assigning Runtime identities."""

    validate_research_task_graph_draft(draft)
    by_role = {node.node_key: node for node in draft.nodes}

    identities = {role: uuid4() for role in RESEARCH_NODE_ROLES}
    nodes_by_role = {
        role: TaskNode(
            node_id=identities[role],
            goal=node.goal,
            dependencies=tuple(
                identities[dependency]
                for dependency in node.dependency_keys
                if not (
                    role == RISK_REVIEW and dependency == NEWS_ANALYSIS
                )
            ),
            expected_output=node.expected_output,
            strategy_id=node.requested_strategy_id,
        )
        for role, node in by_role.items()
    }
    initial_roles = tuple(
        role for role in RESEARCH_NODE_ROLES if role != NEWS_ANALYSIS
    )
    return ResearchTaskDefinition(
        company=company,
        initial_graph=DynamicTaskGraph(
            nodes=tuple(nodes_by_role[role] for role in initial_roles)
        ),
        nodes=MappingProxyType(nodes_by_role),
    )


def validate_research_task_graph_draft(draft: TaskGraphDraft) -> None:
    """Apply Research-domain constraints without assigning Runtime identity."""

    by_role = {node.node_key: node for node in draft.nodes}
    expected = set(RESEARCH_NODE_ROLES)
    actual = set(by_role)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"research graph roles differ; missing={missing}, extra={extra}"
        )
    for role, node in by_role.items():
        expected_strategy = _STRATEGY_BY_ROLE[role]
        if node.requested_strategy_id != expected_strategy:
            raise ValueError(
                f"research role '{role}' requires strategy '{expected_strategy}'"
            )
        dependencies = set(node.dependency_keys)
        required = _REQUIRED_DEPENDENCIES.get(role, set())
        if not required.issubset(dependencies):
            missing = sorted(required - dependencies)
            raise ValueError(
                f"research role '{role}' lacks dependencies: {missing}"
            )
        if role != RISK_REVIEW and NEWS_ANALYSIS in dependencies:
            raise ValueError(
                "only risk_review may depend on dynamically discovered news"
            )
    if by_role[COMPANY_RESEARCH].dependency_keys:
        raise ValueError("company_research must be a root node")


def build_research_task_from_effect(
    company: str,
    effect: PlanningGraphEffect,
) -> ResearchTaskDefinition:
    """Adopt the exact Runtime-owned graph effect after Governance Apply."""

    bindings = {item.node_key: item.node_id for item in effect.node_bindings}
    if set(bindings) != set(RESEARCH_NODE_ROLES):
        raise ValueError("planning graph effect does not cover Research roles")
    graph_nodes = {node.node_id: node for node in effect.graph.nodes}
    nodes_by_role = {
        role: graph_nodes[node_id] for role, node_id in bindings.items()
    }
    return ResearchTaskDefinition(
        company=company,
        initial_graph=effect.graph,
        nodes=MappingProxyType(nodes_by_role),
    )


def build_research_mutations_from_draft(
    definition: ResearchTaskDefinition,
    draft: GraphMutationProposalDraft,
) -> tuple[GraphMutation, ...]:
    """Convert a validated symbolic proposal into bounded Runtime mutations."""

    additions = tuple(
        operation
        for operation in draft.operations
        if operation.kind is GraphMutationOperationKind.ADD_NODE
    )
    dependencies = tuple(
        operation
        for operation in draft.operations
        if operation.kind is GraphMutationOperationKind.ADD_DEPENDENCY
    )
    if len(additions) != 1 or len(dependencies) != 1:
        raise ValueError(
            "research mutation proposal requires one node and one dependency"
        )
    addition = additions[0]
    proposed_node = addition.node
    if proposed_node is None or proposed_node.node_key != NEWS_ANALYSIS:
        raise ValueError("research mutation proposal may add only news_analysis")
    if proposed_node.requested_strategy_id != RESEARCH_STRATEGY_ID:
        raise ValueError("news_analysis requires the research strategy")
    if set(proposed_node.dependency_keys) != {COMPANY_RESEARCH}:
        raise ValueError("news_analysis must depend only on company_research")
    dependency = dependencies[0]
    if (
        dependency.node_key != RISK_REVIEW
        or dependency.dependency_key != NEWS_ANALYSIS
    ):
        raise ValueError(
            "research mutation may only make risk_review depend on news_analysis"
        )
    canonical_news = definition.node(NEWS_ANALYSIS)
    news = TaskNode(
        node_id=canonical_news.node_id,
        goal=proposed_node.goal,
        dependencies=(definition.node(COMPANY_RESEARCH).node_id,),
        expected_output=proposed_node.expected_output,
        strategy_id=proposed_node.requested_strategy_id,
    )
    return (
        GraphMutation.add_node(news, reason=addition.reason),
        GraphMutation.add_dependency(
            definition.node(RISK_REVIEW).node_id,
            news.node_id,
            reason=dependency.reason,
        ),
    )
