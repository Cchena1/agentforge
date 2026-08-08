# AgentForge 依赖评审与高星项目调研

> 快照日期：**2026-08-08**。GitHub 元数据保存在 [research-snapshot.json](research-snapshot.json)。星数和活跃度会变化，不能作为唯一选型依据。

## 1. 调研结论

| 项目 | 快照星数 | 最近推送（UTC） | 许可证快照 | 借鉴/使用内容 | 决策 |
|---|---:|---|---|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | 39,216 | 2026-08-08 | MIT | 图状态机、async node、循环边界、持久化扩展点 | **直接使用库** |
| [Mem0](https://github.com/mem0ai/mem0) | 62,812 | 2026-08-07 | Apache-2.0 | 分层记忆、命名空间、语义召回模式 | 借鉴模式；当前用已有 SQLite/Embedding 实现最小闭环 |
| [Docling](https://github.com/docling-project/docling) | 64,417 | 2026-08-08 | MIT | OCR、版面、表格结构解析 | **可选依赖**，按需安装 |
| [Qdrant](https://github.com/qdrant/qdrant) | 33,851 | 2026-08-08 | Apache-2.0 | HNSW、payload filter、异步客户端 | **可选依赖**；内存模式 PoC 已通过 |
| [LiteLLM](https://github.com/BerriAI/litellm) | 55,867 | 2026-08-08 | GitHub API 未断言 | 路由、retry/fallback/cooldown 模式 | 借鉴模式，不引入；避免再加一层网关和许可证不确定性 |
| [Pydantic AI](https://github.com/pydantic/pydantic-ai) | 19,142 | 2026-08-08 | MIT | schema-first agent、结构化依赖和结果 | 借鉴模式；项目已直接依赖 Pydantic + LangGraph |
| [Instructor](https://github.com/567-labs/instructor) | 13,701 | 2026-08-08 | MIT | Pydantic 校验失败后的结构化重试模式 | 借鉴模式；当前已有轻量修复 + 复验，不重复引包 |
| [LlamaIndex](https://github.com/run-llama/llama_index) | 51,456 | 2026-08-07 | MIT | RAG ingestion、retriever/reranker 分层 | 借鉴模块边界，不引入完整框架 |
| [Haystack](https://github.com/deepset-ai/haystack) | 26,144 | 2026-08-07 | Apache-2.0 | 异步 pipeline 并行分支 | 借鉴 DAG 并行模式，不引入第二套编排框架 |
| [AutoGen](https://github.com/microsoft/autogen) | 60,312 | 2026-04-15 | CC-BY-4.0 快照 | 多 Agent 消息协作 | 未满足“近一月有推送”的优先条件，不复制代码；只作为概念对照 |

除 AutoGen 外，上表重点候选在快照时均有近一个月内推送。`updated_at` 可能只代表仓库元数据变化，因此评审以 `pushed_at` 为维护活跃度证据。

## 2. 为什么没有整仓复制

用户要求优先复用成熟方案。本实现采用“**直接依赖稳定库 + 借鉴已验证边界与策略**”，没有把第三方仓库源码复制进项目：

1. LangGraph、Pydantic、Tenacity、OpenAI SDK、aiosqlite、pypdf 已覆盖核心能力，复制同类代码会增加维护和安全负担。
2. Docling/Qdrant 通过 optional extras 接入，不把重依赖强加给默认 Demo。
3. LiteLLM、Pydantic AI、Instructor、LlamaIndex、Haystack 与当前栈存在功能重叠；先复用模式，避免两套状态机/校验器/检索框架争夺状态所有权。
4. 所有借鉴均为通用架构模式，无第三方源码的逐字复制，因此不引入额外 NOTICE 或源代码许可证传播要求。

若未来确需复制第三方实现，必须记录：上游 commit、文件、许可证、修改差异、安全扫描和更新策略。

## 3. 当前运行依赖能力清单

版本以 `uv.lock` 和本次已创建虚拟环境为准。

| 依赖 | 锁定解析版本 | 已有能力 | 许可证元数据 | 引入理由 |
|---|---:|---|---|---|
| FastAPI | 0.141.1 | async HTTP、OpenAPI、依赖/生命周期 | MIT | 已有 Web 框架，保留 |
| LangGraph | 1.2.10 | 状态图、异步节点、可演进 checkpoint | MIT | 满足显式 Agent 状态机与止损 |
| OpenAI Python | 2.53.0 | AsyncOpenAI、function calling | Apache-2.0 | 已有模型 SDK，避免自写 HTTP 协议 |
| Pydantic | 2.13.4 | 严格模型、JSON Schema、字段约束 | MIT | 数据契约和边界校验核心 |
| pydantic-settings | 2.15.0 | 环境变量配置、SecretStr | MIT | 替代散落配置读取 |
| Tenacity | 9.1.4 | async retry、指数退避/jitter | Apache-2.0 | 成熟重试库，不自写循环 |
| aiosqlite | 0.22.1 | 异步 SQLite/WAL | MIT classifier | 短期/长期记忆与本地索引 |
| pypdf | 6.15.0 | 文本型 PDF 提取和页码 | BSD-3-Clause | 默认轻量 PDF 路径 |
| httpx | 0.28.1 | 异步 HTTP 基础依赖/测试 | BSD-3-Clause | SDK/FastAPI 生态已有能力 |
| Uvicorn | 0.52.1 | ASGI server | BSD-3-Clause | 服务运行 |
| Qdrant Client（optional） | 1.19.0 | async HNSW 服务接入、filter | Apache-2.0 | 大规模向量索引扩展 |
| Docling（optional） | `>=2.50,<3` | OCR、版面、表格结构 | MIT（上游仓库） | 扫描件/复杂表格；未安装验证 |

开发依赖：pytest 9.1.1、pytest-asyncio 1.4.0、Ruff 0.16.2、mypy 1.20.2。

## 4. 新增依赖评审

### LangGraph

- **替代方案**：手写 while-loop、LlamaIndex/Haystack/Pydantic AI 工作流。
- **选择原因**：已明确要求 LangChain/LangGraph；图状态和停止条件比自由循环更易测试；MIT；快照时活跃。
- **风险**：框架升级可能改变 state/checkpoint API。
- **控制**：封装在 `graph.py`，HTTP、工具和存储不直接依赖图内部类型；主版本范围 `<2`。

### Tenacity

- **替代方案**：OpenAI SDK 内置重试、手写循环。
- **选择原因**：需要在多模型路由上统一错误分类、等待策略和日志；关闭 SDK 重试后避免双重放大。
- **风险**：错误分类错误会导致不必要重试。
- **控制**：只重试暂态异常；最大尝试和超时有上限。

### aiosqlite

- **替代方案**：同步 sqlite3 放线程池、PostgreSQL。
- **选择原因**：当前 Demo 无外部基础设施；与 asyncio 主链路一致；WAL 足够做本地 PoC。
- **风险**：单机写并发和跨实例一致性有限。
- **控制**：存储通过模块边界隔离；生产迁移前定义持久化格式和迁移窗口。

### pypdf + Docling optional

- **替代方案**：只用 Docling、Unstructured。
- **选择原因**：普通 PDF 使用轻依赖；扫描件/复杂表格按需启用成熟 Docling，降低默认安装体积。
- **风险**：Docling 模型下载、运行资源和解析差异尚未在目标机验证。
- **控制**：明确 parser/warnings；空扫描文本不伪装为成功。

### Qdrant Client optional

- **替代方案**：SQLite exact scan、FAISS、pgvector。
- **选择原因**：需要 HNSW、过滤和服务化扩展时使用；客户端支持异步；Apache-2.0。
- **风险**：外部服务部署、索引参数与版本兼容。
- **控制**：默认仍为 SQLite；通过 `VectorStore` Protocol 隔离；已完成 in-memory API PoC，未做性能验证。

## 5. 安全与维护记录

- 真实密钥只通过 `SecretStr`/环境变量进入，`.env` 被忽略。
- 文件工具在解析后的工作区根目录内做路径约束，阻止目录穿越。
- Pydantic `extra=forbid` 拒绝模型偷偷加入未定义参数。
- 依赖使用上限版本范围并锁定到 `uv.lock`；更新时先在分支运行测试和静态检查。
- 本次未运行独立 SCA/CVE 扫描，也未审计所有传递依赖漏洞；这是上线前未验证项。
- 建议后续接入 Dependabot/Renovate、OSV-Scanner 或 pip-audit，并给 optional extras 单独建 CI job。

## 6. 复核周期

每月或重大升级前更新 `research-snapshot.json`，复核：最近推送、最新 release、许可证变化、公开安全公告、未解决高优先级 issue、Python 版本支持和 API 弃用。仓库星数只用于发现候选，不作为通过评审的充分条件。
