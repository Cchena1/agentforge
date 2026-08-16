# 数据来源与 200 页分配

## 官方来源

| Benchmark | 官方仓库 | 官方数据页 | 本项目用途 |
|---|---|---|---|
| OmniDocBench | [GitHub](https://github.com/opendatalab/OmniDocBench) | [Hugging Face](https://huggingface.co/datasets/opendatalab/OmniDocBench) | 长期复杂 PDF Parser 回归来源；本轮未重复运行 |
| OHR-Bench | [GitHub](https://github.com/opendatalab/OHR-Bench) | [Hugging Face](https://huggingface.co/datasets/opendatalab/OHR-Bench) | 本轮 200 页 Retrieval、噪声鲁棒性与 Citation Contract |

相关论文：[OHR-Bench: A Comprehensive Benchmark for Optical-to-Text Document Retrieval](https://arxiv.org/abs/2412.02592)。

> 本仓库不重新分发官方大型数据集。复现前应重新检查官方许可证、使用条款和数据版本。

## 本轮分配

| Cohort | 页数 | 比例 | 选择目的 |
|---|---:|---:|---|
| OHR-Bench Longitudinal | 150 | 75% | 与首轮完全相同的确定性页面，进行公平 Before/After 对照 |
| OHR-Bench Unseen Holdout | 50 | 25% | 从后续固定官方窗口选择、与旧样本不重叠，检查泛化 |
| 合计 | 200 | 100% | 固定工作量，不执行压力测试 |

Holdout 的领域配额为 Academic 8、Administration 7、Finance 8、Law 7、Manual 7、News 7、Textbook 6。选择规则不是随机采样：按固定窗口和稳定顺序选择具有 QA 标注的页面，具体 `row_idx`、文档名、页号和内容 Hash 见 `manifests/sample_inventory.json`。

## 本地数据与 Git 边界

原始数据与可重建中间产物位于：

```text
<repository-root>/state/benchmark-200p/
<repository-root>/state/benchmark-200p-v3-holdout/
```

两处目录均由根 `.gitignore` 忽略，包含官方数据窗口、含正文 Manifest、QA、SQLite 索引等。Git 仅保留：

- 官方来源链接与选择规则；
- 不含正文的 200 页样本清单；
- 每个文本 Variant 的 SHA-256；
- 逐查询排名、汇总指标和审阅结果；
- 可复核证据 Hash。

## 数据安全

下载文档属于不可信输入。Benchmark 文本不得修改 System Prompt、Tool 权限、Tenant/ACL 或索引版本策略。若官方数据发生漂移，应重建本地状态并更新 Hash，不能沿用旧结论。
