"""Deterministic Capability Providers for the offline Research demo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Mapping

from pydantic import JsonValue

from adaptive_agent_runtime.tool_ecosystem import (
    Capability,
    CapabilityResolver,
    DeterministicToolSelector,
    ExactCapabilityMatcher,
    InMemoryCapabilityCatalog,
    InMemoryToolRegistry,
    InMemoryToolTraceSink,
    ManagedToolExecutor,
    ToolInvocation,
    ToolProviderMetadata,
    ToolProviderResult,
)

from applications.research_agent.report import build_research_report


INFORMATION_RETRIEVAL = "information_retrieval"
DOCUMENT_ANALYSIS = "document_analysis"
CALCULATION = "calculation"
REPORT_GENERATION = "report_generation"

COMPANY_PROVIDER = "fixture.company_information"
INDUSTRY_PROVIDER = "fixture.industry_information"
COMPETITOR_PROVIDER = "fixture.competitor_information"
NEWS_PRIMARY_PROVIDER = "fixture.news_primary"
NEWS_BACKUP_PROVIDER = "fixture.news_backup"
DOCUMENT_PROVIDER = "fixture.financial_document"
CALCULATION_PROVIDER = "fixture.financial_calculation"
REPORT_PROVIDER = "fixture.markdown_report"


ProviderHandler = Callable[[ToolInvocation], ToolProviderResult]


class FixtureToolProvider:
    """A local provider: deterministic data, no direct API dependency."""

    module_id = "research_agent.provider.fixture"

    def __init__(self, provider_id: str, handler: ProviderHandler) -> None:
        self.provider_id = provider_id
        self._handler = handler
        self.invocations: list[ToolInvocation] = []

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        self.invocations.append(invocation)
        return self._handler(invocation)


@dataclass(frozen=True)
class ResearchToolStack:
    catalog: InMemoryCapabilityCatalog
    registry: InMemoryToolRegistry
    resolver: CapabilityResolver
    selector: DeterministicToolSelector
    trace_sink: InMemoryToolTraceSink
    executor: ManagedToolExecutor
    providers: Mapping[str, FixtureToolProvider]


def _arguments(invocation: ToolInvocation) -> Mapping[str, JsonValue]:
    return invocation.arguments


def _company(invocation: ToolInvocation) -> str:
    value = _arguments(invocation).get("company", "Unknown Company")
    return str(value)


def _company_information(invocation: ToolInvocation) -> ToolProviderResult:
    company = _company(invocation)
    return ToolProviderResult.ok(
        output={
            "company": company,
            "business": "Technology-enabled products and services",
            "research_scope": (
                "financials, industry, competition, material news, and risks"
            ),
            "source": "deterministic company fixture",
        }
    )


def _industry_information(invocation: ToolInvocation) -> ToolProviderResult:
    return ToolProviderResult.ok(
        output={
            "company": _company(invocation),
            "industry": "technology and electrification",
            "outlook": "structurally growing with cyclical pricing pressure",
            "drivers": ["adoption", "cost curves", "regulation"],
        }
    )


def _competitor_information(invocation: ToolInvocation) -> ToolProviderResult:
    return ToolProviderResult.ok(
        output={
            "company": _company(invocation),
            "peers": ["Peer Alpha", "Peer Beta", "Peer Gamma"],
            "differentiation": (
                "Brand and software integration support differentiation, while "
                "price competition limits near-term visibility."
            ),
        }
    )


def _news_primary(invocation: ToolInvocation) -> ToolProviderResult:
    del invocation
    return ToolProviderResult.failed(
        error="primary news fixture is temporarily unavailable",
        retryable=False,
    )


def _news_backup(invocation: ToolInvocation) -> ToolProviderResult:
    return ToolProviderResult.ok(
        output={
            "company": _company(invocation),
            "signal": "mixed",
            "catalyst": "Product execution may improve growth, subject to demand.",
            "controversy": "Pricing and regulatory headlines remain volatile.",
            "source": "deterministic backup news fixture",
        }
    )


def _financial_document(invocation: ToolInvocation) -> ToolProviderResult:
    return ToolProviderResult.ok(
        output={
            "company": _company(invocation),
            "currency": "USD",
            "current": {
                "revenue": 100.0,
                "operating_income": 12.0,
                "net_debt": 8.0,
            },
            "prior": {"revenue": 88.0},
            "source": "normalized illustrative financial statement",
        }
    )


def _calculation(invocation: ToolInvocation) -> ToolProviderResult:
    raw = _arguments(invocation).get("financials")
    financials = raw if isinstance(raw, Mapping) else {}
    current_raw = financials.get("current")
    prior_raw = financials.get("prior")
    current = current_raw if isinstance(current_raw, Mapping) else {}
    prior = prior_raw if isinstance(prior_raw, Mapping) else {}
    def number(value: object) -> float:
        return float(value) if isinstance(value, (int, float)) else 0.0

    revenue = number(current.get("revenue", 0.0))
    prior_revenue = number(prior.get("revenue", 0.0))
    operating_income = number(current.get("operating_income", 0.0))
    net_debt = number(current.get("net_debt", 0.0))
    if revenue <= 0.0 or prior_revenue <= 0.0:
        return ToolProviderResult.failed(
            error="financial inputs require positive current and prior revenue",
            retryable=False,
        )
    return ToolProviderResult.ok(
        output={
            "revenue_growth_pct": round(
                ((revenue / prior_revenue) - 1.0) * 100.0,
                2,
            ),
            "operating_margin_pct": round(
                (operating_income / revenue) * 100.0,
                2,
            ),
            "net_debt_to_revenue": round(net_debt / revenue, 2),
            "calculation_basis": "illustrative normalized financials",
        }
    )


def _report(invocation: ToolInvocation) -> ToolProviderResult:
    arguments = _arguments(invocation)
    raw_analyses = arguments.get("analyses")
    analyses = raw_analyses if isinstance(raw_analyses, Mapping) else {}
    raw_preferences = arguments.get("preferences")
    preferences = (
        tuple(raw_preferences)
        if isinstance(raw_preferences, (list, tuple))
        else ()
    )
    report = build_research_report(
        _company(invocation),
        analyses,
        preferences,
    )
    return ToolProviderResult.ok(output=report.model_dump(mode="json"))


def build_research_tool_stack() -> ResearchToolStack:
    """Register capabilities and fixture Providers in the Runtime ecosystem."""

    catalog = InMemoryCapabilityCatalog()
    for capability in (
        Capability(
            capability_id=INFORMATION_RETRIEVAL,
            name="Information Retrieval",
            description="Retrieve company, industry, competitor, and news facts.",
            tags=("research", "financial"),
        ),
        Capability(
            capability_id=DOCUMENT_ANALYSIS,
            name="Document Analysis",
            description="Normalize financial statement documents.",
            tags=("research", "document"),
        ),
        Capability(
            capability_id=CALCULATION,
            name="Calculation",
            description="Calculate deterministic financial indicators.",
            tags=("research", "financial"),
        ),
        Capability(
            capability_id=REPORT_GENERATION,
            name="Report Generation",
            description="Generate a structured Markdown research report.",
            tags=("research", "markdown"),
        ),
    ):
        catalog.register(capability)

    providers = {
        COMPANY_PROVIDER: FixtureToolProvider(
            COMPANY_PROVIDER,
            _company_information,
        ),
        INDUSTRY_PROVIDER: FixtureToolProvider(
            INDUSTRY_PROVIDER,
            _industry_information,
        ),
        COMPETITOR_PROVIDER: FixtureToolProvider(
            COMPETITOR_PROVIDER,
            _competitor_information,
        ),
        NEWS_PRIMARY_PROVIDER: FixtureToolProvider(
            NEWS_PRIMARY_PROVIDER,
            _news_primary,
        ),
        NEWS_BACKUP_PROVIDER: FixtureToolProvider(
            NEWS_BACKUP_PROVIDER,
            _news_backup,
        ),
        DOCUMENT_PROVIDER: FixtureToolProvider(
            DOCUMENT_PROVIDER,
            _financial_document,
        ),
        CALCULATION_PROVIDER: FixtureToolProvider(
            CALCULATION_PROVIDER,
            _calculation,
        ),
        REPORT_PROVIDER: FixtureToolProvider(
            REPORT_PROVIDER,
            _report,
        ),
    }
    registry = InMemoryToolRegistry(catalog)
    registrations = (
        (COMPANY_PROVIDER, INFORMATION_RETRIEVAL, ("company", "privileged"), 100),
        (INDUSTRY_PROVIDER, INFORMATION_RETRIEVAL, ("industry",), 100),
        (COMPETITOR_PROVIDER, INFORMATION_RETRIEVAL, ("competitor",), 100),
        (NEWS_PRIMARY_PROVIDER, INFORMATION_RETRIEVAL, ("news", "primary"), 100),
        (NEWS_BACKUP_PROVIDER, INFORMATION_RETRIEVAL, ("news", "backup"), 50),
        (DOCUMENT_PROVIDER, DOCUMENT_ANALYSIS, ("financial_document",), 100),
        (CALCULATION_PROVIDER, CALCULATION, ("financial_metrics",), 100),
        (REPORT_PROVIDER, REPORT_GENERATION, ("report", "markdown"), 100),
    )
    for provider_id, capability_id, tags, priority in registrations:
        registry.register(
            ToolProviderMetadata(
                provider_id=provider_id,
                name=provider_id.replace("fixture.", "").replace("_", " ").title(),
                capability_id=capability_id,
                description="Deterministic offline Research demo Provider.",
                input_schema={"type": "object"},
                tags=tags,
                selection_priority=priority,
            ),
            providers[provider_id],
        )
    trace_sink = InMemoryToolTraceSink()
    resolver = CapabilityResolver(
        catalog=catalog,
        registry=registry,
        matcher=ExactCapabilityMatcher(),
    )
    executor = ManagedToolExecutor(registry=registry, trace_sink=trace_sink)
    return ResearchToolStack(
        catalog=catalog,
        registry=registry,
        resolver=resolver,
        selector=DeterministicToolSelector(),
        trace_sink=trace_sink,
        executor=executor,
        providers=providers,
    )
