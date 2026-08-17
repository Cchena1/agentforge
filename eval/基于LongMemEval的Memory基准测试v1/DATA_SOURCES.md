# 数据来源与许可证

| 来源 | 官方地址 | 本地固定版本 | 许可证 | 本轮使用 |
|---|---|---|---|---|
| LongMemEval | https://github.com/xiaowu0162/LongMemEval | `9e0b455f4ef0e2ab8f2e582289761153549043fc`（2026-05-11） | MIT | Cleaned S 中五类各 10 个固定 Question ID |
| LongMemEval Cleaned Data | https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned | 本地 `longmemeval_s_cleaned.json` | 以数据集页面和上游声明为准 | 仅本地流式读取，不上传 |
| AgentForge | 当前仓库 | 当前工作树 | 项目许可证 | 20 个隔离、更新删除、摘要与外置 Fixture |

## 数据选择

五类固定为：`information_extraction`、`multi_session`、`knowledge_update`、`temporal_reasoning`、`abstention`，每类按官方文件稳定顺序选择前 10 个满足条件的样本。样本清单只保存 Question ID、分类和问题 SHA-256。

## 本地数据策略

- 约 277 MB 的官方 JSON 只保存在 `state/benchmark-sources/`；`state/` 已由 `.gitignore` 排除。
- Benchmark 目录不保存问题、会话、答案、向量数据库或临时 SQLite。
- 运行时以增量 Decoder 流式读取，减少一次性内存占用；本轮未进行资源压测。
