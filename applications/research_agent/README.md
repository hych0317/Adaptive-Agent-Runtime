# Research Agent Application

Research Agent 是 Adaptive Agent Runtime 的金融研究展示应用，不属于 Runtime Core。它只通过 Runtime 的公开接口组合 Orchestration、Tool、Context-Memory、Evaluation、Governance 和 LLM Layer。

> 本应用用于展示 Agent Runtime 架构，不提供投资建议，也不是实时行情或专业金融数据系统。

## 两种运行模式

| 模式 | 信息来源 | 适用场景 |
|---|---|---|
| `LLM Research` | Information Retrieval 通过所选 LLM 生成结构化信息，最终报告也由 LLM 生成 | 展示真实模型调用、Capability 路由、Tool Trace 和模型切换 |
| `Fixture Demo` | 使用内置确定性 Provider，并执行基线运行与当前运行 | 稳定复现 Context & Memory 生命周期、失败重试、Evaluation、Governance Review 和 Optimization Proposal |

Web 的“自动”模式会在存在活动 LLM Target 时优先选择 `LLM Research`；没有活动 LLM 时使用 `Fixture Demo`。用户也可以显式切换模式。

LLM Research 默认使用简体中文完成规划、节点分析、风险审视和报告生成；公司、产品名称及通用财务缩写可以保留英文。

LLM Research 当前属于“模型信息综合”，不是联网检索：

- 不声称访问了实时网页、新闻、财报或行情。
- Observation 的 `source` 为 `llm model synthesis (not live retrieval)`。
- 新闻输出必须携带时效性说明和不确定性。
- 财务文档示例与计算仍保留确定性实现，避免把模型生成的数字伪装成可靠财务数据。

## Runtime 调用链

```text
User Task
  -> Research Application
  -> Dynamic Task Graph
  -> Research Strategy
  -> Capability Requirement
  -> Tool Registry / Provider Selection
  -> LLM Information Provider 或 Fixture Provider
  -> Tool Observation
  -> Context Update / Memory Interaction
  -> Report Strategy
  -> Evaluation
  -> Governance
  -> Research Report
```

LLM Information Provider 仍遵守以下边界：

```text
Information Retrieval Capability
  -> LLM Provider
  -> Managed Tool Executor
  -> Tool Observation
```

Strategy 不直接调用模型，Tool Provider 不直接修改 Context 或 Memory。

### LLM Tool Intent

支持 Reasoning 的 API Target 会把 Runtime 允许的候选能力契约暴露给 LLM。LLM 只能提出 Intent，不能直接执行工具：

```text
Runtime Candidate Capability
  -> LLM Tool Intent
  -> Runtime Schema / Task Boundary Validation
  -> Capability Matching / Provider Selection
  -> Governance / Tool Execution
  -> Tool Observation
```

Web 的“评估与治理”页会显示候选能力、Intent 参数、实际 Provider 和 Observation。候选能力可用不代表模型每次都必须调用；已有证据充分时模型可以直接完成分析。当前 `deepseek-research` API Target 默认允许每个分析回合最多提出一次 Intent。Codex CLI 后端当前只提供生成能力，因此不会在界面中伪装成支持 Tool Intent。

## 启动 Web Console

在项目根目录运行：

```powershell
python examples/research_web.py
```

访问 `http://127.0.0.1:8765`。

推荐操作：

1. 打开右上角“LLM 设置”。
2. 选择 Target 并刷新 Provider 实时模型列表。
3. 选择模型，保存并执行连接验证。
4. 运行模式保持“自动”，或显式选择 `LLM Research`。
5. 输入“分析 Tesla 投资价值”等研究任务。

修改 Python 服务代码后需要重启 Web 服务；静态资源会直接从磁盘读取，但 Python 路由和 Application Composition 只在进程启动时加载。

## LLM 配置

公开部署配置位于：

```text
config/llm.toml
```

本地私有配置位于：

```text
config/llm.local.toml
```

私有配置由 Runtime 的 `TOMLProviderConfigRepository` 管理，可保存：

- Provider API Key
- Target 对应的模型选择
- 可选 reasoning effort

API Key 不会返回浏览器，也不会进入 Runtime Trace。Codex CLI Target 使用启动 Web 服务的宿主用户 OAuth 会话。

## CLI

确定性演示：

```powershell
python examples/research_demo.py
```

指定 LLM 配置运行：

```powershell
python examples/research_demo.py --llm-config config/llm.toml --llm-target codex-research
```

配置 LLM 后，CLI 使用 LLM Research 信息流程；未配置 LLM 时执行 Fixture Demo。

## 输出内容

Web Console 展示：

- Dynamic Task Graph 与实时节点状态
- Runtime Execution Trace
- Capability、Provider 与 Tool Observation
- Context 生命周期和 Memory 交互
- 结构化研究报告
- Outcome、Trajectory 和 Component Evaluation
- Failure Pattern 与 Optimization Proposal
- Tool、Memory、Graph Mutation 和 Optimization Governance
- Runtime 候选能力、LLM Tool Intent、Provider 选择与执行结果

判断信息来源时应查看 Tool Observation 的 `provider` 和 `source`，不要使用 Strategy 名称或页面顶部模型标签代替证据来源。

## 主要模块

```text
applications/research_agent/
├── agent.py            # Application composition root
├── capabilities.py     # Fixture 与 LLM Tool Providers
├── cognition.py        # 可选 LLM cognitive capability bindings
├── strategies.py       # Research / Report / Review strategies
├── tasks.py            # Research Task Graph definition
├── llm_deployment.py   # LLM Target composition
├── web.py              # Local Web API and SSE
├── web_llm.py          # Local LLM settings adapter
└── web_assets/         # Lightweight console UI
```

## 验证

```powershell
$env:PYTHONPATH = "src;."
python -m unittest tests.research_agent.test_research_agent
python -m unittest tests.research_agent.test_web
```

全项目回归：

```powershell
$env:PYTHONPATH = "src;."
python -m unittest discover -s tests
```
