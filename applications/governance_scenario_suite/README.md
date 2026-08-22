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

凭据只通过环境变量读取，不能写入场景 YAML 或命令参数：

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
