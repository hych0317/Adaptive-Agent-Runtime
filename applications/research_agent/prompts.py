"""Provider-neutral prompt templates kept at the Application boundary."""

DEFAULT_RESEARCH_LANGUAGE = "zh-CN"
CHINESE_OUTPUT_INSTRUCTION = (
    "默认使用简体中文输出研究内容；公司、产品名称和通用财务缩写可保留英文。"
)

RESEARCH_PROMPT = (
    "收集 {company} 的研究证据，明确区分事实、假设与信息局限。"
)

RISK_REVIEW_PROMPT = (
    "独立审视 {company} 的投资论点，列出主要风险以及可能使论点失效的条件。"
)

REPORT_PROMPT = (
    "为 {company} 生成简洁的 Markdown 研究报告，涵盖证据、财务分析、行业背景、"
    "竞争对手比较、风险与结论。"
)
