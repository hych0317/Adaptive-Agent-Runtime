"""Deterministic Capability Providers for the offline Research demo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, cast

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
    ToolProvider,
    ToolInvocation,
    ToolProviderMetadata,
    ToolProviderResult,
)
from adaptive_agent_runtime.llm import (
    ArtifactGenerationCapability,
    CapabilityInvocationMetadata,
    CapabilityTurnKind,
    GenerationRequest,
    InferenceCorrelation,
)

from applications.research_agent.prompts import CHINESE_OUTPUT_INSTRUCTION
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
LLM_COMPANY_PROVIDER = "llm.company_information"
LLM_INDUSTRY_PROVIDER = "llm.industry_information"
LLM_COMPETITOR_PROVIDER = "llm.competitor_information"
LLM_NEWS_PROVIDER = "llm.news_information"

PRIVILEGED_INFORMATION_PROVIDERS = frozenset(
    {COMPANY_PROVIDER, LLM_COMPANY_PROVIDER}
)


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


class LLMInformationProvider:
    """Application Tool Provider backed by a governed LLM capability."""

    module_id = "research_agent.provider.llm_information"

    def __init__(
        self,
        *,
        provider_id: str,
        topic: str,
        instruction: str,
        output_schema: Mapping[str, JsonValue],
        generator: ArtifactGenerationCapability,
    ) -> None:
        self.provider_id = provider_id
        self._topic = topic
        self._instruction = instruction
        self._output_schema = output_schema
        self._generator = generator
        self.invocations: list[ToolInvocation] = []

    async def invoke(self, invocation: ToolInvocation) -> ToolProviderResult:
        self.invocations.append(invocation)
        requested_at = datetime.now(timezone.utc).isoformat()
        turn = await self._generator.generate(
            GenerationRequest(
                instruction=(
                    self._instruction
                    + " "
                    + CHINESE_OUTPUT_INSTRUCTION
                    + " Use model knowledge only. Do not claim live browsing, "
                    "real-time market data, or access to documents that were not "
                    "provided. State material uncertainty in caveats."
                ),
                context={
                    "company": _company(invocation),
                    "research_topic": self._topic,
                    "requested_at": requested_at,
                },
                media_type="application/json",
                output_schema=self._output_schema,
            ),
            invocation=CapabilityInvocationMetadata(
                correlation=InferenceCorrelation(
                    run_id=invocation.correlation.run_id,
                    task_id=invocation.correlation.task_id,
                    node_id=invocation.correlation.node_id,
                    action_id=(
                        invocation.correlation.action_id
                        or invocation.invocation_id
                    ),
                ),
                trace_attributes={
                    "application": "research_agent",
                    "operation": "information.retrieve.llm",
                    "provider_id": self.provider_id,
                },
            ),
        )
        if turn.kind is CapabilityTurnKind.TOOL_INTENT or turn.result is None:
            return ToolProviderResult.failed(
                error="LLM information Provider returned no structured result",
                retryable=False,
            )
        if turn.result.media_type != "application/json":
            return ToolProviderResult.failed(
                error="LLM information Provider returned the wrong media type",
                retryable=False,
            )
        if not isinstance(turn.result.content, Mapping):
            return ToolProviderResult.failed(
                error="LLM information Provider output must be an object",
                retryable=False,
            )
        output = dict(turn.result.content)
        output["source"] = "llm model synthesis (not live retrieval)"
        output["requested_at"] = requested_at
        return ToolProviderResult.ok(output=output)


@dataclass(frozen=True)
class ResearchToolStack:
    catalog: InMemoryCapabilityCatalog
    registry: InMemoryToolRegistry
    resolver: CapabilityResolver
    selector: DeterministicToolSelector
    trace_sink: InMemoryToolTraceSink
    executor: ManagedToolExecutor
    providers: Mapping[str, ToolProvider]


def _arguments(invocation: ToolInvocation) -> Mapping[str, JsonValue]:
    return cast(Mapping[str, JsonValue], invocation.arguments)


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


def _object_schema(
    properties: Mapping[str, JsonValue],
    required: tuple[str, ...],
) -> Mapping[str, JsonValue]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": dict(properties),
        "required": list(required),
    }


_STRING: dict[str, JsonValue] = {"type": "string", "minLength": 1}
_STRING_LIST: dict[str, JsonValue] = {
    "type": "array",
    "minItems": 1,
    "items": _STRING,
}

_LLM_INFORMATION_SPECS: tuple[
    tuple[
        str,
        str,
        str,
        tuple[str, ...],
        Mapping[str, JsonValue],
    ],
    ...,
] = (
    (
        LLM_COMPANY_PROVIDER,
        "company profile",
        "Build a concise investment-research profile for the company.",
        ("company", "privileged", "llm"),
        _object_schema(
            {
                "company": _STRING,
                "business": _STRING,
                "research_scope": _STRING,
                "key_facts": _STRING_LIST,
                "caveats": _STRING_LIST,
            },
            ("company", "business", "research_scope", "key_facts", "caveats"),
        ),
    ),
    (
        LLM_INDUSTRY_PROVIDER,
        "industry analysis",
        "Summarize the company's industry, outlook, structural drivers, and risks.",
        ("industry", "llm"),
        _object_schema(
            {
                "company": _STRING,
                "industry": _STRING,
                "outlook": _STRING,
                "drivers": _STRING_LIST,
                "caveats": _STRING_LIST,
            },
            ("company", "industry", "outlook", "drivers", "caveats"),
        ),
    ),
    (
        LLM_COMPETITOR_PROVIDER,
        "competitor analysis",
        "Identify relevant competitors and summarize differentiation and risks.",
        ("competitor", "llm"),
        _object_schema(
            {
                "company": _STRING,
                "peers": _STRING_LIST,
                "differentiation": _STRING,
                "competitive_risks": _STRING_LIST,
                "caveats": _STRING_LIST,
            },
            (
                "company",
                "peers",
                "differentiation",
                "competitive_risks",
                "caveats",
            ),
        ),
    ),
    (
        LLM_NEWS_PROVIDER,
        "news and catalyst analysis",
        (
            "Summarize material news themes and catalysts known to the model, "
            "while making recency limits explicit."
        ),
        ("news", "primary", "llm"),
        _object_schema(
            {
                "company": _STRING,
                "signal": _STRING,
                "catalyst": _STRING,
                "controversy": _STRING,
                "as_of": _STRING,
                "caveats": _STRING_LIST,
            },
            (
                "company",
                "signal",
                "catalyst",
                "controversy",
                "as_of",
                "caveats",
            ),
        ),
    ),
)


def build_research_tool_stack(
    *,
    llm_information_generator: ArtifactGenerationCapability | None = None,
) -> ResearchToolStack:
    """Register fixture and optional LLM Providers in the Runtime ecosystem."""

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

    providers: dict[str, ToolProvider] = {
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
    registrations: list[
        tuple[str, str, tuple[str, ...], int]
    ] = [
        (COMPANY_PROVIDER, INFORMATION_RETRIEVAL, ("company", "privileged"), 100),
        (INDUSTRY_PROVIDER, INFORMATION_RETRIEVAL, ("industry",), 100),
        (COMPETITOR_PROVIDER, INFORMATION_RETRIEVAL, ("competitor",), 100),
        (NEWS_PRIMARY_PROVIDER, INFORMATION_RETRIEVAL, ("news", "primary"), 100),
        (NEWS_BACKUP_PROVIDER, INFORMATION_RETRIEVAL, ("news", "backup"), 50),
        (DOCUMENT_PROVIDER, DOCUMENT_ANALYSIS, ("financial_document",), 100),
        (CALCULATION_PROVIDER, CALCULATION, ("financial_metrics",), 100),
        (REPORT_PROVIDER, REPORT_GENERATION, ("report", "markdown"), 100),
    ]
    if llm_information_generator is not None:
        for provider_id, topic, instruction, tags, schema in _LLM_INFORMATION_SPECS:
            providers[provider_id] = LLMInformationProvider(
                provider_id=provider_id,
                topic=topic,
                instruction=instruction,
                output_schema=schema,
                generator=llm_information_generator,
            )
            registrations.append(
                (provider_id, INFORMATION_RETRIEVAL, tags, 200)
            )
    for provider_id, capability_id, tags, priority in registrations:
        is_llm = provider_id.startswith("llm.")
        registry.register(
            ToolProviderMetadata(
                provider_id=provider_id,
                name=(
                    provider_id.removeprefix("fixture.")
                    .removeprefix("llm.")
                    .replace("_", " ")
                    .title()
                ),
                capability_id=capability_id,
                description=(
                    "LLM-backed Research information Provider."
                    if is_llm
                    else "Deterministic offline Research demo Provider."
                ),
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
