"""Runtime-owned Tool contracts that may be exposed to cognitive providers."""

from __future__ import annotations

from adaptive_agent_runtime.llm import ToolSpecification

from applications.research_agent.capabilities import INFORMATION_RETRIEVAL


RESEARCH_RETRIEVAL_SCOPES = (
    "company",
    "industry",
    "competitor",
    "news",
)

RESEARCH_INFORMATION_RETRIEVAL_TOOL = ToolSpecification(
    capability_id=INFORMATION_RETRIEVAL,
    name="retrieve_research_evidence",
    description=(
        "Propose retrieval of bounded company, industry, competitor, or news "
        "evidence. Adaptive Agent Runtime validates and executes the proposal."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "company": {"type": "string", "minLength": 1},
            "scope": {"type": "string", "enum": list(RESEARCH_RETRIEVAL_SCOPES)},
            "query": {"type": "string", "minLength": 1},
        },
        "required": ["company", "scope"],
        "additionalProperties": False,
    },
)
