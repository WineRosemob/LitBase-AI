# Workflow

## End-to-end pipeline | 端到端流程

1. Define a research topic.
2. Expand the query.
3. Search multiple metadata sources.
4. Merge and deduplicate results.
5. Enrich OA metadata.
6. Apply rule-based scoring.
7. Optionally apply embedding and LLM rubric scoring.
8. Select candidate papers.
9. Download OA PDFs or use authorized institutional access.
10. Export tables, cards, BibTeX, JSONL, and reports.

## Practical sequence | 实操顺序

1. Run `pytest`.
2. Run `litbase-ai doctor`.
3. Run a small `litbase-ai search` with low `--limit`.
4. Inspect `outputs/demo`.
5. Scale up only after the small run looks correct.

## Failure handling | 故障处理

- If tests fail, fix code or mocks first.
- 如果测试失败，先修代码或 mock。
- If `doctor` fails on external services, inspect whether it is missing configuration, a network issue, or an API-side problem.
- 如果 `doctor` 因外部服务失败，要区分是配置缺失、网络问题还是 API 侧问题。
- If a search run produces no PDFs, inspect `search_diagnostics.json` and `run.log`.
- 如果没有下载到 PDF，优先查看 `search_diagnostics.json` 和 `run.log`。
