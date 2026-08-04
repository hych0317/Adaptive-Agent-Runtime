# 砚台：个人知识库应用

砚台是构建在 Adaptive Agent Runtime 之上的本地单用户应用。它把“来源存档”和“个人知识”严格分开：网页原文、视频时间戳转录、Agent 草稿和完整审阅对话进入存档区；只有用户在可编辑确认界面提交的最终文本，才会进入知识库和默认检索范围。

## 运行

在仓库根目录安装 Runtime 后启动：

```powershell
pip install -e .
python -m applications.personal_knowledge.web --database data/knowledge.sqlite3
```

浏览器访问 `http://127.0.0.1:8775`。服务默认只监听回环地址；该端口与 Research Application 的 `8765` 分离。

左侧“模型设置”读取同一工作区的 `config/llm.toml` target 目录。可在界面中选择 Provider target、读取完整模型下拉列表、保存模型覆盖/API Key。“测试并激活”会探测当前表单中的 target 与模型，连接成功后直接激活；失败时保留原模型。API Key 只写入被 Git 忽略的 `config/llm.local.toml`，不会由设置接口返回。

也可以启动时直接选择：

```powershell
python -m applications.personal_knowledge.web `
  --database data/knowledge.sqlite3 `
  --llm-config config/llm.toml `
  --llm-target deepseek-research
```

数据被刻意分为两个 SQLite 文件：

- `knowledge.sqlite3`：来源文本、转录、审阅对话、知识条目、版本、分类、订阅和 FTS 索引。
- `runtime.sqlite3`：Runtime Context/Memory/Governance 等运行态数据。个人偏好使用这里的 Memory，不把知识正文写入 Memory。

## 核心流程

1. 用户粘贴想法、文本或 URL，或由本地订阅调度器触发采集。
2. 来源通过 Runtime Tool Capability 执行；网页使用 `source.fetch_text`，视频使用 `video.extract_transcript`。
3. Agent 生成 `KnowledgeProposal`，它只能保存在审阅存档，不能直接写知识表。
4. 用户在前端确认弹窗编辑标题、内容分类、正文、标签和引用。
5. 提交的精确载荷经过 Runtime Governance 指纹绑定和单次授权，在 apply point 写入 SQLite。
6. 默认 FTS 只搜索已确认知识；“同时搜索存档”必须由用户主动开启。问答只能引用本次检索实际返回的证据键。

已有知识的编辑会开启新的审阅会话。数据库保留当前版本与最多两个历史版本。删除进入可恢复回收站。

## Runtime LLM 接入

命令行未配置模型时使用离线确定性实现，便于无密钥启动和测试；它不会获得额外写入权限。正式部署时在应用组合根注入 Runtime 的 `ArtifactGenerationCapability`：

```python
from adaptive_agent_runtime.llm import (
    compose_managed_capabilities,
    compose_managed_inference,
)
from applications.personal_knowledge.web import PersonalKnowledgeApplication

inference = compose_managed_inference((configured_backend,))
capabilities = compose_managed_capabilities(inference.gateway)
app = PersonalKnowledgeApplication(
    "data/knowledge.sqlite3",
    generation_capability=capabilities.generator,
    memory_extraction_capability=capabilities.extractor,
)
```

同一个 Runtime generation capability 同时用于知识提案和带引用问答。模型只产生 authority-free draft，不能绕过确认边界。

## 视频配置

视频 Provider 遵循 Hermes `video-summary` 的 transcript-first 流程：先 probe，再严格使用脚本返回的动态超时运行 ASR；读取 `transcript_path` 与 JSONL 时间戳片段；长内容分块覆盖；绝不根据标题、简介或评论生成摘要。

应用会自动查找同级 `Hermes/hermes-data/skills/media/video-summary/scripts/fetch_video_transcript.py`。也可通过启动参数显式指定（优先级最高）：

```powershell
python -m applications.personal_knowledge.web `
  --database data/knowledge.sqlite3 `
  --video-script "C:\path\to\fetch_video_transcript.py"
```

或通过环境变量覆盖：

```powershell
$env:PERSONAL_KNOWLEDGE_VIDEO_SCRIPT="C:\path\to\fetch_video_transcript.py"
python -m applications.personal_knowledge.web --database data/knowledge.sqlite3
```

解析顺序为 `--video-script`、`PERSONAL_KNOWLEDGE_VIDEO_SCRIPT`、同级 Hermes 仓库。媒体和音频只存在于每次调用的临时目录，成功、失败或超时后都会清理。数据库只保存视频链接、元数据、全文转录和时间戳片段。如果 helper 仍未找到，视频 Tool 会返回明确配置错误，并在服务终端保留完整 traceback。

## 自动订阅

订阅调度器属于应用层，不是 Runtime Tool，也不依赖 Codex 自动化。它定时调用已注册的来源 Tool，比对文本哈希，有变化时只创建待审提案。

服务运行期间会启动轻量进程内调度器。也可由系统计划任务调用一次性命令：

```powershell
python -m applications.personal_knowledge.web --database data/knowledge.sqlite3 --poll-once
```

这是有意保持简单的 MVP 示例：网页/频道 URL 按整个可读文本检测变化，不承诺覆盖所有平台的登录、风控或复杂 feed 语义。

## Memory 边界

- 用户明确表达的行为偏好可作为受治理 `MemoryCandidate` 直接固化。
- Agent 通过 Runtime `MemoryExtractionCapability` 推断偏好，只返回 `PreferenceSuggestion`；前端会另行询问，用户确认前 Memory Store 保持不变。
- Memory key 固定使用 `personal_knowledge.preference.*` 命名空间，条件限定为本应用。
- 来源正文、转录、知识条目和对话不会进入 Memory。

## 验证

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests\personal_knowledge -v

Remove-Item Env:PYTHONPATH
$env:MYPYPATH="src"
python -m mypy applications\personal_knowledge --strict
```

定向测试覆盖：未经确认不入库、精确编辑载荷、陈旧确认拒绝、分类确认、版本与回收站、FTS 存档隔离、视频 Tool/时间戳转录、订阅变化检测、Memory 推断确认边界，以及真实 localhost HTTP 端到端流程。
