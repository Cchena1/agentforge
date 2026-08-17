# 基于 LongMemEval 的 Memory 基准测试 v1

该目录保存 AgentForge 的首轮 Memory 工程评估，固定 **70 项**：LongMemEval 五类分层子集 50、AgentForge 隔离与压缩对抗集 20。

## 快速入口

- [完整报告](report.md)
- [数据来源与许可证](DATA_SOURCES.md)
- [证据链与限制](EVIDENCE.md)
- [逐项结果](results/cases.json)
- [汇总指标](results/summary.json)
- [去文本化样本清单](manifests/sample_inventory.json)
- [证据 Hash 清单](evidence_manifest.json)

## 复现

```powershell
.\.venv\Scripts\python.exe "eval\基于LongMemEval的Memory基准测试v1\scriptsun_benchmark.py"
.\.venv\Scripts\python.exe "eval\基于LongMemEval的Memory基准测试v1\scriptsdversarial_audit.py"
```

复现需要本地 `state/benchmark-sources/longmemeval/data/longmemeval_s_cleaned.json`。该文件由官方仓库指向的 Hugging Face Dataset 获取，仅保存在 Git Ignore 的 `state/`。
