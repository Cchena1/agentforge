# 基于 OmniDocBench 和 OHR-Bench 的 RAG 基准测试 v3

本目录归档 AgentForge 于 **2026-08-16** 完成的第三次 200 页 RAG Benchmark。V3 专门验证前两轮暴露的**检索召回率不足**：150 页沿用首轮 OHR-Bench 确定性样本，保证纵向可比；50 页使用与前者不重叠的 OHR-Bench 固定窗口作为 Unseen Holdout，检查修复是否只对旧样本有效。

> Benchmark Suite 仍以 OmniDocBench 与 OHR-Bench 为长期数据来源。由于本次代码只修改 Retrieval，不修改 PDF Parser，200 页预算全部投入 OHR-Bench；OmniDocBench Parser 证据继续保留在 V1，不把未变化链路重复计入本轮工作量。

## 快速结论

| 指标 | 200 页综合结果 |
|---|---:|
| 不重复页面 | 200 |
| 查询数 | 559 / Variant |
| Ground Truth Recall@5 | 95.53% |
| Formatting Noise Recall@5 | 95.53% |
| Semantic/OCR Noise Recall@5 | 88.91% |
| Hit-level Citation Validity | 100.00% |
| 压力测试 | 未执行 |

纵向 150 页同集对照中，三个 Variant 的 Recall@5 分别从 **73.30% / 70.26% / 57.38%** 提升到 **94.38% / 94.38% / 86.42%**。这证明本次 BM25 召回修复解决了原 HashEmbedding 错误主导 Fusion 的主要缺陷，但不代表所有领域、所有 Vector Store 或答案级引用均已达到生产标准。

## 目录

```text
基于OmniDocBench和OHR-Bench的RAG基准测试v3/
├── README.md
├── report.md
├── DATA_SOURCES.md
├── EVIDENCE.md
├── evidence_manifest.json
├── manifests/
│   └── sample_inventory.json
├── results/
│   ├── adversarial_gate.json
│   ├── cohort_summary.json
│   ├── environment.json
│   ├── ohr_retrieval_gt_text.json
│   ├── ohr_retrieval_formatting_noise_moderate.json
│   ├── ohr_retrieval_semantic_noise_MinerU_moderate.json
│   └── ohr_retrieval_summary.json
└── scripts/
    ├── adversarial_audit.py
    └── generate_evidence_manifest.py
```

## 校验命令

```powershell
.\.venv\Scripts\python.exe "eval\基于OmniDocBench和OHR-Bench的RAG基准测试v3\scriptsdversarial_audit.py"
.\.venv\Scripts\python.exe "eval\基于OmniDocBench和OHR-Bench的RAG基准测试v3\scripts\generate_evidence_manifest.py"
```

官方大数据集、正文、问题、答案、页面图片、PDF 和 SQLite 索引仍只保存在 Git Ignore 的 `state/` 中，不随 Git 分发。
