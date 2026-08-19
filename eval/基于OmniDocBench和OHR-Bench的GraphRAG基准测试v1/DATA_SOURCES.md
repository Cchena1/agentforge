# 数据来源与 100 页分配

## 官方来源

| Benchmark | 官方仓库 | 官方数据页 | 本轮用途 |
|---|---|---|---|
| OmniDocBench | [GitHub](https://github.com/opendatalab/OmniDocBench) | [Hugging Face](https://huggingface.co/datasets/opendatalab/OmniDocBench) | 继续作为复杂 PDF Parser 长期回归来源；本轮 Parser 未变化，不重复消耗页面预算 |
| OHR-Bench | [GitHub](https://github.com/opendatalab/OHR-Bench) | [Hugging Face](https://huggingface.co/datasets/opendatalab/OHR-Bench) | 本轮 100 页 GraphRAG Retrieval、Multi-evidence 与噪声鲁棒性对照 |

相关论文：[OHR-Bench: A Comprehensive Benchmark for Optical-to-Text Document Retrieval](https://arxiv.org/abs/2412.02592)。

> 本仓库不重新分发官方大型数据集。复现前应重新检查官方许可证、数据版本和使用条款。

## 页面选择

| 领域 | 页数 |
|---|---:|
| Academic | 15 |
| Administration | 14 |
| Finance | 15 |
| Law | 14 |
| Manual | 14 |
| News | 14 |
| Textbook | 14 |
| 合计 | 100 |

确定性选择规则：

1. 纳入 200 页历史 Pool 中全部 22 条 Multi-evidence Query 对应的 27 个页面；
2. 纳入第三轮 Semantic/OCR Noise Recall@5 失败 Query 对应的 31 个页面；
3. 两者合并去重后为 56 页；
4. 其余 44 页优先选择相同文档的邻近候选、Unseen Holdout、QA 数较多页面，并按 `sample_id` 稳定排序补齐领域配额。

最终清单见 `manifests/sample_inventory.json`，不含正文。

## 本地状态

可重建的大型/敏感产物位于：

```text
<repository-root>/state/benchmark-200p/
<repository-root>/state/benchmark-200p-v3-holdout/
<repository-root>/state/benchmark-graphrag-100p/
```

以上目录由根 `.gitignore` 忽略。不得将官方正文、问题答案、PDF、图片、Embedding 或 SQLite Index 提交到 Git。
