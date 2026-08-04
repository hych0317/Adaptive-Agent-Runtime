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
        "提议补充检索边界明确的公司、行业、竞争对手或新闻证据。"
        "Adaptive Agent Runtime 将校验并执行该提议。"
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
