# Benchmark 证据索引

> 生成日期：2026-08-15
> 原则：保留可复核的轻量证据，不提交官方大型数据集、实际状态库、完整 Trace 或工具二进制。

## 1. 机器可读结果

| 文件 | 用途 |
|---|---|
| `results/operational_benchmark.json` | 200 页预算、Ingestion、恢复前后检索、Metrics/Trace、Backup/Restore 和 11 项 acceptance |
| `results/metrics_snapshot.prom` | Prometheus 文本快照；包含必需 metric family 与持久化 Backup/Restore 状态 |
| `results/trace_summary.json` | Span 数量、名称、Trace ID 数量、内容安全检查与完整 Trace 文件哈希 |
| `results/backup_manifest_sanitized.json` | 脱敏备份清单；保留文件 size/hash/kind，不提交 SQLite 数据 |
| `results/alert_validation.json` | promtool/amtool 版本与配置、规则、规则单测验证结果 |
| `results/cli_recovery_validation.json` | CLI backup、verify、隔离 restore 演练结果；路径已脱敏 |
| `evidence_manifest.json` | 本目录关键证据与 gitignored 源 manifest 的 SHA-256 汇总 |

## 2. 可复现脚本

- `scripts/run_operational_benchmark.py`：固定 200 页顺序运维闭环测试。
- `scripts/validate_alerts.py`：调用官方 promtool/amtool，并将结果规范化为 JSON。

## 3. 未提交但可本地复核的内容

- `state/benchmark-200p/`：OmniDocBench 50 页与 OHR-Bench 150 页原始/派生资源；由 `.gitignore` 排除。
- `state/benchmark-observability-200p/`：canonical 文本、SQLite 状态、完整 Trace、备份和隔离恢复目录；由 `.gitignore` 排除。
- `state/tools/`：promtool/amtool 官方二进制；由 `.gitignore` 排除。

源数据来源、选择比例与重新获取方式见 `DATA_SOURCES.md`。本次没有提交超过 1 GB 的内容，也没有执行压测。
