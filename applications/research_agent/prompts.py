"""Provider-neutral prompt templates kept at the Application boundary."""

RESEARCH_PROMPT = (
    "Collect evidence for {company}; separate facts, assumptions, and limitations."
)

RISK_REVIEW_PROMPT = (
    "Independently challenge the investment thesis for {company} and list the "
    "conditions that would invalidate it."
)

REPORT_PROMPT = (
    "Create a concise Markdown report for {company} with evidence, financial "
    "analysis, industry context, competitor comparison, risks, and conclusion."
)

