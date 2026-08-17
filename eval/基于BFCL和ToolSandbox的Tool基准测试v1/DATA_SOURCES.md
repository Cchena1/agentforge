# 数据来源与许可证

| 来源 | 官方地址 | 本地固定版本 | 许可证 | 本轮使用 |
|---|---|---|---|---|
| BFCL | https://github.com/ShishirPatil/gorilla | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`（2026-03-23） | Apache-2.0 | 30 个固定 ID：Single、Multiple、Parallel、Irrelevance、Multi-turn |
| ToolSandbox | https://github.com/apple/ToolSandbox | `165848b9a78cead7ca7fe7c89c688b58e6501219`（2025-11-06） | Apple 项目自带许可证 | 15 个能力投影：Stateful、Canonicalization、Insufficient Information |
| AgentForge | 当前仓库 | 当前工作树 | 项目许可证 | 15 个内部对抗样本 |

## 本地数据策略

- 官方源码和 Long-lived Raw Data 仅保存在 `state/benchmark-sources/`，该目录已被 `.gitignore` 排除。
- Git 仅保存去文本化样本 ID/Hash、模型 Tool Call、评分结果、报告和证据 Hash。
- ToolSandbox 未安装到 AgentForge 主虚拟环境。其官方依赖固定 OpenAI 1.17、Pydantic 2.7、LangChain 0.1 和较重 NLP 栈，与当前运行时存在冲突风险。
- 因此本轮 ToolSandbox 是能力投影，不是官方 ExecutionContext + Milestone/Snapshot 完整评分；该边界在全部报告中显式保留。
