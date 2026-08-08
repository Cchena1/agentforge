# AgentForge 验收证据矩阵

> 证据基线：2026-08-08。本次明确 **不进行压测**；并行测试只验证调度语义和时间重叠，不施加持续负载。

## 1. 需求到实现/测试的对应关系

| 验收项 | 实现证据 | 测试/文档证据 | 状态 |
|---|---|---|---|
| Retry / Fallback | `llm.ModelGateway` + Tenacity；多 `ModelRoute`；本地安全工具与人工升级 | `test_retry_recovers_from_transient_timeout`；`test_backward_compatible_chat_contract_and_offline_fallback`；架构 §3 | 已实现 |
| 合理的记忆系统 | `MemoryStore`：SQLite WAL、会话摘要、向量化长期记忆、namespace | `test_short_term_memory_is_isolated_and_compacted`；`test_long_term_memory_uses_semantic_namespace`；架构 §7 | 已实现 |
| 全异步调用策略 | AsyncOpenAI、aiosqlite、async LangGraph node、asyncio timeout/gather/semaphore/lock | 工具并行、依赖、资源锁测试；mypy strict | 已实现 |
| asyncio 异步编程 | `AsyncToolExecutor`、`MultiAgentCoordinator`、所有 I/O 服务 async | `test_independent_tools_run_concurrently_without_load_testing` | 已实现 |
| RAG 策略 | parser → semantic chunker → embedding → vector store → hybrid rerank → citation | `test_rag_semantic_chunks_and_returns_citations`；架构 §9 | 已实现 |
| LangChain / LangGraph | `AgentGraph` 使用 LangGraph `StateGraph` | `test_langgraph_stops_repeated_tool_loop` | 已实现 |
| Function Calling | Pydantic 输入模型生成 JSON Schema；OpenAI tool definition `strict=true` | `test_schema_first_tool_contract_rejects_wrong_types_and_extra_fields` | 已实现 |
| 工具描述避免瞎选 | PURPOSE/USE ONLY WHEN/DO NOT USE WHEN/SIDE EFFECTS/RETURNS 模板 | `test_tool_descriptions_explain_selection_boundaries`；架构 §4 | 已实现 |
| 参数 Schema 粒度 | `extra=forbid`、required、枚举/范围/模式、`additionalProperties=false` | Schema 边界测试；架构 §4 | 已实现 |
| JSON 缺字段/类型错误兜底 | JSON 提取/轻量修复一次，再用 Pydantic 复验；失败切备用路由/人工 | `test_structured_output_missing_field_is_repaired_and_revalidated`；架构 §3 | 已实现 |
| 重试、备用工具或人工的选择 | 错误分类决策树；确定性错误不盲重试；风险/不可恢复错误人工升级 | 架构 §3；API `degraded/handoff_required/error_code` | 已实现 |
| 多工具并行依赖与冲突 | DAG `depends_on`、`$ref`、环检测、semaphore、resource lock、上游失败 skip | `test_dependencies_resolve_results_and_cycles_are_rejected`；`test_shared_resource_lock_prevents_conflicting_mutation` | 已实现 |
| 多 Agent 上下文隔离 | 每 Agent 独立 `AgentContext/private_context/memory_namespace` | `test_multi_agent_context_isolated_and_result_compressed` | 已实现基础层 |
| 多 Agent 结果压缩 | `AgentResult` 仅摘要/事实/引用/产物/告警，长度上限 | 同上；架构 §6 | 已实现 |
| 记忆分层 | 工作、短期、长期、知识四层；状态所有权分离 | memory tests；架构 §7 | 已实现 |
| 短期上下文存储 | 最近消息 + 字符/条数预算 + 滚动摘要 | `test_short_term_memory_is_isolated_and_compacted` | 已实现 |
| 长期记忆与向量联动 | 写入 embedding，按 namespace 语义召回并结合 importance | `test_long_term_memory_uses_semantic_namespace` | 已实现 |
| 死循环步数止损 | `max_agent_steps` + 重复工具签名哈希上限 | `test_langgraph_stops_repeated_tool_loop`；架构 §8 | 已实现 |
| 扫描件与表格 | 扫描 PDF 检测并提示 OCR；Docling optional；CSV/TSV 行结构和页码/locator | 三个 RAG parser tests；架构 §9 | 默认检测/表格已验证；Docling 未安装验证 |
| 固定长度还是语义切片 | 标题/段落/表格行优先，字符预算和 overlap 兜底 | RAG tests；架构 §9 | 已实现 |
| Embedding 选型差异 | hash 仅离线基线；托管/本地模型按质量、成本、延迟、隐私评测 | 架构 §9 | 策略已定义；未做质量 benchmark |
| 向量索引选择 | 默认 SQLite exact；可选 Qdrant HNSW/filters | `test_qdrant_adapter_in_memory_round_trip`；依赖评审 | 两条路径功能已验证；远程 Qdrant 未验证 |
| 检索延迟 | 每次检索返回 `latency_ms`；方法规定 p50/p95/p99 等 | 架构 §10 | 观测已实现；按要求未压测 |
| 重排序 | 向量 + 词法融合 + 重复 chunk 去除；保留各阶段分数 | RAG citation test；架构 §9 | 已实现轻量版 |
| 最终带来源引用 | `Citation` 契约，RAG 工具/ChatResponse 汇总并生成引用尾注 | `test_rag_semantic_chunks_and_returns_citations` | 已实现 |
| Pydantic 首道拦截 | API、模型计划、工具参数/返回、记忆、检索全部严格模型 | schema test、structured output test、FastAPI 422 路径测试 | 已实现 |
| 类型注解 | `mypy --strict` 覆盖 `src/agent_service` | mypy 通过 | 已实现 |
| 虚拟环境与依赖管理 | `pyproject.toml`、`uv.lock`、dependency groups、optional extras、`.gitignore` | `uv sync --all-groups --extra vector` 成功；依赖评审 | 已实现 |

## 2. API 兼容证据

- 原有 `POST /chat` 仍接受 `message/history`。
- 原有响应 `reply` 保留。
- 原有 `tool_calls[].name/arguments/result` 保留。
- 新字段增量加入：`tool_name/status/latency_ms/error_code/citations/session_id/degraded/handoff_required`。
- `test_backward_compatible_chat_contract_and_offline_fallback` 对无在线模型场景进行验证。
- 路径逃逸由 `test_path_escape_is_rejected` 验证。

## 3. 已执行命令与结果

```text
uv sync --all-groups --extra vector
Resolved 167 packages; qdrant-client 1.19.0 installed.

.venv\Scripts\ruff.exe check src main.py tests --fix
All checks passed!

.venv\Scripts\mypy.exe src\agent_service
Success: no issues found in 15 source files

.venv\Scripts\pytest.exe
17 passed, 1 warning

目标仓库最终复验（CPython 3.12.13）：Ruff 通过，mypy strict 通过，17 tests passed；Node.js 与 PowerShell 语法检查通过。
```

警告来自 FastAPI/Starlette TestClient 对 httpx 兼容层的弃用提示，不是业务测试失败；详见“已知限制”。

## 4. 未执行/未验证

- **任何压力测试、吞吐测试、长时间并发测试：按用户要求未执行。**
- 没有真实模型 API Key，因此没有验证真实供应商的 function calling 方言、限流和备用模型切换。
- 没有安装 Docling 和 OCR 模型，因此只验证扫描件检测与 fallback 告警，未验证真实 OCR/复杂表格质量。
- Qdrant 只用客户端内存模式做 CRUD/检索 PoC，未连接远程服务、未验证索引参数与持久化。
- 没有业务标注集，因此未比较不同 Embedding/reranker 的 Recall@k、MRR、nDCG 和引用正确率。
- 没有运行 SCA/CVE、容器、部署、跨进程 SQLite 竞争或多实例一致性测试。
- 目标仓库旧 `.venv` 的 Python home 曾含乱码；已于 2026-08-08 用已验证的 CPython 3.12.13 执行 `uv venv --clear` 和 `uv sync --all-groups --extra vector` 重建，且普通终端可启动。

## 5. 已知限制与后续门槛

1. SQLite 向量检索会扫描候选行，语料明显增长后应切 Qdrant/pgvector 并在获准后评测。
2. 轻量线性 rerank 不是 cross-encoder；上线前需用业务查询集验证质量收益。
3. 确定性摘要可控但信息压缩有限；如改成模型摘要，必须增加事实保真和 prompt injection 防护。
4. 多 Agent 当前是库级协调器，尚无公开 API、权限模型、预算或冲突仲裁器。
5. 长期记忆缺少删除/导出/保留期/PII 策略；进入真实用户环境前必须补齐。
6. Starlette TestClient 的 httpx 兼容路径有弃用警告，应跟踪 FastAPI/Starlette 推荐迁移到 httpx2 的时间窗口。
