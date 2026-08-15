# Benchmark 数据来源与复现边界

## 1. 官方来源

| Benchmark | 官方代码仓库 | 官方数据页 | 本轮用途 |
|---|---|---|---|
| OmniDocBench | [GitHub](https://github.com/opendatalab/OmniDocBench) | [Hugging Face](https://huggingface.co/datasets/opendatalab/OmniDocBench) | 复杂版面 PDF 解析入口与失败分类 |
| OHR-Bench | [GitHub](https://github.com/opendatalab/OHR-Bench) | [Hugging Face](https://huggingface.co/datasets/opendatalab/OHR-Bench) | Chunking、检索、噪声鲁棒性和 Citation Contract |

相关论文：

- [OHR-Bench: A Comprehensive Benchmark for Optical-to-Text Document Retrieval](https://arxiv.org/abs/2412.02592)

> 本项目只记录来源和复现信息，不重新分发官方大型数据集。下载前应以官方仓库和数据页面的最新许可证、使用条款及内容权利说明为准。

## 2. 本轮实际使用的源文件

| 文件 | 获取位置 | SHA-256 |
|---|---|---|
| `OmniDocBench.json` | OmniDocBench Hugging Face 数据页 | `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496` |
| `ohr_qas_v2.json` | OHR-Bench Hugging Face 数据页 | `2446db28741fa9f392067ee7aae7f3b05e0d85c584069a50ddd5b1b5bc783f58` |

下载和页面读取接口由 `scripts/prepare_samples.py` 固化：

```text
https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/main/images/{image_name}?download=true
https://datasets-server.huggingface.co/rows?dataset=opendatalab%2FOHR-Bench&config=default&split=train&offset={offset}&length={length}
```

这些地址是复现入口，不代表本仓库拥有或重新授权其中内容。

## 3. 200 页分配与确定性规则

| 来源 | 页面数 | 比例 | 选择目的 |
|---|---:|---:|---|
| OmniDocBench | 50 | 25% | 复杂版面、表格、公式、多栏和视觉页面解析 |
| OHR-Bench | 150 | 75% | 领域覆盖、检索、噪声退化和引用结构 |
| 合计 | 200 | 100% | 固定总工作量 |

抽样不是随机抽样，而是按稳定排序后的分层规则选择，因此不依赖随机种子。具体样本身份、分层、官方页面名、页面 Hash 或 OHR 行号/页号保存在 `manifests/sample_inventory.json`。该清单故意删除正文、问题、答案和噪声文本，只保留复核所需元数据。

## 4. 本地数据布局

复现时的所有大文件和中间产物写入：

```text
<repository-root>/state/benchmark-200p/
```

该目录由项目根 `.gitignore` 忽略，典型内容包括：

- 官方 JSON 数据；
- OmniDocBench 页面图片及临时单页 PDF；
- 含原文的完整 Manifest；
- SQLite 索引及 sidecar 文件；
- Parser 原始输出和模型缓存；
- 可重新生成的中间文件。

这些内容是本地复现输入或临时产物，不是 Git 交付物。

## 5. 数据最小化理由

第一性原则是：Benchmark 结论需要能够验证，但验证不等于复制全部上游数据。当前提交采用以下证据链：

```mermaid
flowchart LR
    A["官方来源 + 源文件 Hash"] --> B["固定抽样规则"]
    B --> C["去文本化 200 页清单"]
    C --> D["汇总指标 + 逐查询排名"]
    D --> E["报告结论"]
```

- **完整性**：源文件 Hash 和样本级 Hash 可识别输入漂移。
- **可审阅性**：逐查询结果保留相关 Source ID、命中 Source ID、首个相关排名和 Citation 校验结果。
- **最小披露**：不提交页面正文、问题、答案、原图和 PDF。
- **可重建性**：脚本和官方入口足以在获准环境中重新下载并生成本地状态。

## 6. 许可与安全提示

- 不把第三方数据集许可证解释为本项目许可证的一部分。
- 不在 CI 中默认下载大型数据或外部模型。
- 下载内容应视为不可信输入；不得允许文档文本改变系统 Prompt、工具权限或索引隔离策略。
- 如官方文件内容或许可证发生变化，应重新计算源 Hash、更新本文件并重跑 Benchmark，而不是沿用旧结论。
