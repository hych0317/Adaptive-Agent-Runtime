# AAR Governance Scenario Suite

这套评测以最终权威状态、外部流水、Audit、模型上下文和 Memory 证据判定结果，不以模型文本是否“看起来合理”为标准。

## 常用命令

```powershell
# Full AAR 确定性套件
python scripts/run_governance_scenarios.py deterministic --profile full_aar

# Plain Agent 与 Full AAR 同条件对比
python scripts/run_governance_scenarios.py compare --profiles plain_agent,full_aar

# 完整发布门禁：Full + 所有精确消融 + 数据库不变性
python scripts/run_governance_scenarios.py gate

# P2 授权篡改 → R1 响应丢失 → M1 跨用户记忆演示
python scripts/run_governance_scenarios.py demo
```

默认报告写入 `artifacts/governance-scenarios/`，也可通过 `--output-dir` 指定目录。JSON 是机器可读事实源，Markdown 从相同的类型化结果生成。

## 真实模型 E2E

凭据不能写入场景 YAML、报告或命令参数。通用 E2E 可通过环境变量运行：

```powershell
$env:AAR_GOVERNANCE_API_KEY = "..."
python scripts/run_governance_scenarios.py e2e `
  --service openai `
  --target-id governance/openai `
  --model-id "YOUR_FIXED_MODEL_ID" `
  --api-key-env AAR_GOVERNANCE_API_KEY `
  --repeat 10
```

E2E 报告将 Provider 故障单独列为 `INCONCLUSIVE`，并展示危险提案、拦截、安全恢复、合法完成和审批的 `x/n` 小样本计数。它不参与普通 PR 的确定性发布门禁，也不能被解释为生产概率或用户价值验证。

本地 `config/llm.toml` 与被 Git 忽略的 `config/llm.local.toml` 已配置时，可运行完整的强制提案故障恢复矩阵：

```powershell
python scripts/run_fault_recovery_pilots.py
```

该矩阵覆盖 P1 跨用户订单、P2 订单/金额/操作三种审批篡改、P4 旧状态版本、P5 不可信商品描述和 C1 超额退款。故障只在模型结构化输出规范化之后、Runtime 治理之前生效；Runtime 拒绝后，恢复上下文保留原任务，但用实际注入提案替换原模型提案。

R1/R2 的故障位于外部提交和对账阶段，M1/M2 的故障位于记忆候选与召回阶段，因此继续由各自的跨进程恢复和记忆作用域 Harness 验证，不伪装成模型提案故障。
