# AgentForge 可观测性与灾备运行手册

> 版本：0.3.x；日期：2026-08-15。默认 SQLite/Local-first 路径适用。所有备份命令都必须在 AgentForge 服务停止后执行。

## 1. 启动与健康检查

```powershell
uv sync --dev
$env:AI_AGENT_METRICS_ENABLED = "true"
$env:AI_AGENT_TRACE_JSONL_ENABLED = "true"
uv run uvicorn agent_service.app:create_app --factory --host 127.0.0.1 --port 8000
```

- Liveness：`GET /health`
- Readiness：`GET /ready`
- Metrics：`GET /metrics`
- 每个可追踪 HTTP 响应返回 `X-Trace-ID`。

## 2. Trace

默认文件：`state/telemetry/traces.jsonl`。默认每个文件最大 20 MiB，保留 5 个历史文件。

```powershell
$env:AI_AGENT_TRACE_JSONL_MAX_BYTES = "20971520"
$env:AI_AGENT_TRACE_JSONL_BACKUP_COUNT = "5"
# 可选：发送到 OTLP/HTTP Collector
$env:AI_AGENT_OTEL_EXPORTER_OTLP_ENDPOINT = "http://127.0.0.1:4318/v1/traces"
```

禁止把 query、Prompt、原文、Chunk 正文、用户消息或 API key 写入 Span attributes。

## 3. Prometheus 与告警

配置位于 `deploy/observability/`。示例假定 AgentForge 在 `127.0.0.1:8000`，Prometheus 在 `9090`，Alertmanager 在 `9093`。

```powershell
.\state\tools\prometheus-3.13.1.windows-amd64\promtool.exe check config .\deploy\observability\prometheus.yml
.\state\tools\prometheus-3.13.1.windows-amd64\promtool.exe test rules .\deploy\observability\rule_tests.yml
.\state\tools\alertmanager-0.32.1.windows-amd64\amtool.exe check-config .\deploy\observability\alertmanager.example.yml
```

`alertmanager.example.yml` 的 webhook 是占位地址，上线前必须替换为企业通知渠道，并完成 firing/resolved 双向演练。

## 4. 备份、验证与恢复

```powershell
# 1) 停止服务后创建备份，并自动校验
uv run python -m agent_service.operations backup --state-dir .\state --output-dir E:\agentforge-backups

# 2) 独立复核已有备份
uv run python -m agent_service.operations verify --backup-dir E:\agentforge-backups\afb_YYYYMMDDTHHMMSSZ_xxxxxxxx

# 3) 先恢复到隔离目录
uv run python -m agent_service.operations restore --backup-dir E:\agentforge-backups\afb_YYYYMMDDTHHMMSSZ_xxxxxxxx --target-state-dir .\state-restored

# 4) 用 state-restored 启动实例并执行健康检查、检索回归；确认后再安排切换
```

不要把唯一备份保存在同一块物理磁盘。建议执行 3-2-1：至少 3 份副本、2 种介质、1 份异地或离线副本。当前项目只实现应用层本地备份；副本调度由部署系统拥有。

## 5. 告警处置矩阵

| 告警 | 首要动作 | 恢复信号 |
|---|---|---|
| `AgentForgeTargetDown` | 检查进程、端口、state lock 和启动日志 | `up == 1` |
| `AgentForgeHighHttp5xxRatio` | 按 `X-Trace-ID` 定位失败 Span | 5xx ratio 回落 |
| `AgentForgeDegradedRetrieval` | 检查 Vector/Lexical 分支和超时 | degraded 增量停止 |
| `AgentForgeIngestionFailure` | 查看 Job failure_code/quality_report，禁止静默索引 | Job completed 或 needs_review 被处理 |
| `AgentForgeBackupNeverSucceeded/Stale` | 停服执行 backup + verify + 隔离 restore drill | status 加载新成功时间 |
| `AgentForgeRestoreVerificationFailed` | 禁止提升恢复目录，检查 hash/integrity | 新一轮完整 restore drill 通过 |
| `AgentForgeTraceExportFailure` | 检查磁盘容量、权限或 OTLP Collector | exporter failure 不再增长 |

## 6. 恢复演练最小验收

1. Manifest Schema 可通过 Pydantic 校验。
2. 每个文件 size、SHA-256、SQLite integrity 均通过。
3. 恢复目录内 Chunk 数、Active Version 数与源状态一致。
4. 固定问题集恢复前后检索 Chunk ID 签名一致。
5. 服务可启动，`/health`、`/ready`、`/metrics` 正常。
6. 失败时不激活恢复目录，不删除源状态。