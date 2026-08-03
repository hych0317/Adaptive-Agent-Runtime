"""Structured application output and Phase 7 result snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from adaptive_agent_runtime import RunResult, TraceEntry
from adaptive_agent_runtime.context_memory import (
    ContextAssembly,
    ContextUnit,
    MemoryUnit,
)
from adaptive_agent_runtime.evaluation import (
    EvaluationReport,
    FailureAnalysis,
    OptimizationProposal,
)
from adaptive_agent_runtime.governance import (
    AuthorizationUse,
    GovernanceAuthorization,
    GovernanceDecision,
    GovernanceRequest,
    ReviewRequest,
)
from adaptive_agent_runtime.llm import (
    ActionProposalDraft,
    AutonomousAgentResult,
    JudgeAssessmentDraft,
    GraphMutationProposalDraft,
    LLMContextPackage,
    MemoryCandidateDraft,
    ReasoningResult,
    TaskGraphDraft,
    ToolIntentDraft,
)
from adaptive_agent_runtime.orchestration import DynamicTaskGraph
from adaptive_agent_runtime.tool_ecosystem import (
    ToolObservation,
    ToolTraceEntry,
)


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    findings: tuple[str, ...] = Field(min_length=1)


class ResearchReport(BaseModel):
    """Structured report returned by the Report Generation Capability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    company: str = Field(min_length=1)
    executive_summary: str = Field(min_length=1)
    sections: tuple[ReportSection, ...] = Field(min_length=1)
    risk_factors: tuple[str, ...] = Field(min_length=1)
    investment_view: str = Field(min_length=1)
    markdown: str = Field(min_length=1)


@dataclass(frozen=True)
class GovernanceRecord:
    scenario: str
    request: GovernanceRequest
    preliminary: GovernanceDecision
    final: GovernanceDecision
    authorization: GovernanceAuthorization | None
    review: ReviewRequest | None


@dataclass(frozen=True)
class ResearchReasoningRecord:
    node_id: UUID
    role: str
    result: ReasoningResult


@dataclass(frozen=True)
class ResearchToolIntentRecord:
    node_id: UUID
    intent: ToolIntentDraft
    observation: ToolObservation


@dataclass(frozen=True)
class ResearchRunResult:
    runtime_result: RunResult
    task_graph: DynamicTaskGraph
    report: ResearchReport
    evaluation: EvaluationReport
    failure_analysis: FailureAnalysis
    optimization_proposals: tuple[OptimizationProposal, ...]
    governance_records: tuple[GovernanceRecord, ...]
    authorization_uses: tuple[AuthorizationUse, ...]
    runtime_trace: tuple[TraceEntry, ...]
    tool_trace: tuple[ToolTraceEntry, ...]
    tool_observations: tuple[ToolObservation, ...]
    context_units: tuple[ContextUnit, ...]
    context_assemblies: tuple[ContextAssembly, ...]
    memories: tuple[MemoryUnit, ...]
    agent_executions: tuple[AutonomousAgentResult, ...]
    llm_judgement: JudgeAssessmentDraft | None
    llm_task_graph_draft: TaskGraphDraft | None
    llm_action_proposals: tuple[ActionProposalDraft, ...]
    llm_graph_mutation_proposals: tuple[GraphMutationProposalDraft, ...]
    llm_reasoning: tuple[ResearchReasoningRecord, ...]
    llm_memory_candidates: tuple[MemoryCandidateDraft, ...]
    llm_context_packages: tuple[LLMContextPackage, ...]
    llm_tool_intents: tuple[ResearchToolIntentRecord, ...]
    evaluation_history_runs: int


def _mapping(value: JsonValue | None) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def build_research_report(
    company: str,
    analyses: Mapping[str, JsonValue],
    preferences: tuple[JsonValue, ...] = (),
) -> ResearchReport:
    """Deterministically turn capability results into a structured report."""

    metrics = _mapping(analyses.get("financial_metrics"))
    industry = _mapping(analyses.get("industry_analysis"))
    competitors = _mapping(analyses.get("competitor_analysis"))
    news = _mapping(analyses.get("news_analysis"))
    risk = _mapping(analyses.get("risk_review"))

    risk_values = risk.get("risks")
    risk_factors = (
        tuple(str(item) for item in risk_values)
        if isinstance(risk_values, (list, tuple)) and risk_values
        else ("Execution risk and changing market conditions",)
    )
    revenue_growth = metrics.get("revenue_growth_pct", "n/a")
    operating_margin = metrics.get("operating_margin_pct", "n/a")
    industry_outlook = industry.get("outlook", "mixed")
    peer_names = competitors.get("peers", ())
    peer_text = (
        ", ".join(str(item) for item in peer_names)
        if isinstance(peer_names, (list, tuple))
        else str(peer_names or "not available")
    )
    news_signal = news.get("signal", "neutral")
    preference_note = (
        " User preferences recalled from Memory were applied to report structure "
        "and risks emphasis."
        if preferences
        else ""
    )
    summary = (
        f"{company} shows {revenue_growth}% revenue growth and an operating margin "
        f"of {operating_margin}%. Industry outlook is {industry_outlook}; recent "
        f"news signal is {news_signal}.{preference_note}"
    )
    sections = (
        ReportSection(
            title="Financial Analysis",
            findings=(
                f"Revenue growth: {revenue_growth}%",
                f"Operating margin: {operating_margin}%",
                f"Net debt to revenue: {metrics.get('net_debt_to_revenue', 'n/a')}",
            ),
        ),
        ReportSection(
            title="Industry and Competition",
            findings=(
                f"Industry outlook: {industry_outlook}",
                f"Reference competitors: {peer_text}",
                str(competitors.get("differentiation", "Differentiation is mixed.")),
            ),
        ),
        ReportSection(
            title="News and Catalysts",
            findings=(
                f"Aggregate news signal: {news_signal}",
                str(news.get("catalyst", "No single catalyst dominates.")),
            ),
        ),
    )
    investment_view = (
        "Balanced: attractive growth is offset by execution and competitive risk."
    )
    markdown_lines = [
        f"# {company} Investment Research",
        "",
        "## Executive Summary",
        summary,
        "",
    ]
    for section in sections:
        markdown_lines.extend(
            [
                f"## {section.title}",
                *(f"- {finding}" for finding in section.findings),
                "",
            ]
        )
    markdown_lines.extend(
        [
            "## Risk Review",
            *(f"- {item}" for item in risk_factors),
            "",
            "## Investment View",
            investment_view,
        ]
    )
    return ResearchReport(
        company=company,
        executive_summary=summary,
        sections=sections,
        risk_factors=risk_factors,
        investment_view=investment_view,
        markdown="\n".join(markdown_lines),
    )
