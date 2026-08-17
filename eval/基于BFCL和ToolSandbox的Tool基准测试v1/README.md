# 基于 BFCL 和 ToolSandbox 的 Tool Orchestration 基准测试 v1

该目录保存 AgentForge 的首轮 Tool Orchestration 工程评估，固定 **60 项**：BFCL 30、ToolSandbox Projection 15、AgentForge 对抗集 15。

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
.\.venv\Scripts\python.exe "eval\基于BFCL和ToolSandbox的Tool基准测试v1\scriptsun_benchmark.py"
.\.venv\Scripts\python.exe "eval\基于BFCL和ToolSandbox的Tool基准测试v1\scriptsdversarial_audit.py"
```

复现需要本地 `state/benchmark-sources/` 中的官方浅克隆，以及可用的 OpenAI-compatible 模型配置。`state/` 已被 Git Ignore，不上传官方数据和密钥。
