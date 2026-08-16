# 测试数据来源

本轮不保存或上传大型官方数据集，仅保留来源说明和固定样本证据。原始数据、PDF 和派生运行态位于 Git ignored 的 `state/benchmark-200p/` 与 `state/benchmark-observability-200p/`。

- OmniDocBench：复用 50 页固定样本，来源、许可证与下载方式见相邻 benchmark 的 `DATA_SOURCES.md` 和 `manifests/sample_inventory.json`。
- OHR-Bench：复用 150 页固定样本，来源、许可证与下载方式同上。
- 本轮输入：从固定页面 manifest 的 ground-truth text 派生 `.txt`，只用于隔离测试运维闭环；不是新的 PDF 数据集，也不替代 PDF Parser 评价。

页面预算不变量：`50 + 150 = 200`，sample_id 必须唯一且正文非空，runner 在执行前强制校验。