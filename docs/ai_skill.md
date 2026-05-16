# AI Skill Guide for LitBase-AI

## Project Goal | 项目目标

LitBase-AI automates the construction of topic-focused literature workspaces.

LitBase-AI 用于自动化构建主题型文献库，核心能力包括：

- multi-source retrieval
- 多源检索
- merge and deduplication
- 文献合并与去重
- rule-based scoring and LLM rubric scoring
- 规则评分与 LLM 评分
- export to Excel, BibTeX, JSONL, and Markdown literature cards
- 导出 Excel、BibTeX、JSONL、Markdown 文献卡片
- compliant open-access PDF download
- 合规下载开放获取 PDF

## Core directories | 核心目录说明

- `litbase_ai/search/`: search clients for OpenAlex, Crossref, arXiv, Semantic Scholar, optional CNKI.
- `litbase_ai/query/`: query expansion and translation helpers.
- `litbase_ai/scoring/`: rule score, evidence score, embedding score, LLM rubric score, and final aggregation.
- `litbase_ai/download/`: OA resolver, downloader orchestration, institutional proxy helpers, legacy sources.
- `litbase_ai/enrich/`: metadata enrichment such as journal ranking and Unpaywall.
- `litbase_ai/storage/`: exporters and literature card generation.
- `litbase_ai/utils/`: progress, logging, healthcheck, cache, and text helpers.
- `tests/`: offline-first unit tests; external service behavior should stay mocked or skipped.
- `examples/`: safe public sample configuration and demo shell snippets.
- `scripts/`: environment setup and demo run helpers.

## Typical run flow | 典型运行流程

1. Install the environment.
2. Copy a config template.
3. Run `doctor`.
4. Run `search` on a small topic.
5. Inspect outputs.

## Common commands | 常用命令

```bash
bash scripts/setup_conda_env.sh
litbase-ai --help
litbase-ai doctor --env-file examples/example.env --output-dir outputs/doctor
litbase-ai init --env-file examples/example.env --non-interactive
litbase-ai search --env-file examples/example.env --topic "climate change integrated assessment model" --limit 20 --year-from 2018 --disable-cnki --output-dir outputs/demo
pytest
python -m compileall litbase_ai
```

## Configuration variables | 配置变量解释

- `OPENALEX_MAILTO`: OpenAlex contact email.
- `UNPAYWALL_EMAIL`: Unpaywall-required email.
- `LLM_API_KEY`: LLM key for query expansion and scoring.
- `LLM_BASE_URL`, `LLM_MODEL`: LLM endpoint and model.
- `SEMANTIC_SCHOLAR_API_KEY`: optional Semantic Scholar support.
- `sentence-transformers`: optional dependency for embedding scoring only.
- `ENABLE_SCIHUB`, `ENABLE_LIBGEN`: legacy/experimental, disabled by default.
- `ENABLE_ARXIV_DOWNLOAD`: legal arXiv download support.
- `ENABLE_EZPROXY`, `ENABLE_INST_PROXY`: optional authorized institutional access.

## Output files | 输出文件解释

- `papers_raw.jsonl`: merged raw paper metadata.
- `papers_scored.xlsx`: scored paper table.
- `papers_selected.xlsx`: selected papers after threshold/top-k logic.
- `papers_cards.xlsx`: literature card spreadsheet.
- `references.bib`: BibTeX export.
- `summary_report.md`: text summary of the run.
- `expanded_query.json`: query expansion artifact.
- `search_diagnostics.json`: diagnostics across search, scoring, and download stages.
- `run.log`: runtime log for the pipeline.
- `pdf/`: downloaded PDFs.
- `literature_cards/`: per-paper Markdown cards.

## AI editing rules | AI 修改项目时的注意事项

- Do not write real API keys, passwords, or personal emails.
- 不要写入真实 API key、密码或私人邮箱。
- Do not commit `.env`, `outputs/`, PDFs, cookies, or caches.
- 不要提交 `.env`、`outputs/`、PDF、cookie 或缓存。
- Do not make non-authorized download paths the default.
- 不要把非授权下载方式设为默认启用。
- If dependencies change, sync `requirements.txt`, `environment.yml`, and `pyproject.toml`.
- 修改依赖后同步 `requirements.txt`、`environment.yml`、`pyproject.toml`。
- If CLI changes, sync README and docs.
- 修改 CLI 后同步 README 和 docs。
- If output fields change, sync exporters and tests.
- 修改输出字段后同步 exporters 和测试。
- If scoring logic changes, sync `litbase_ai/config/scoring.yaml` and related tests.
- 修改评分逻辑后同步 `litbase_ai/config/scoring.yaml` 和相关测试。

## Debugging order | 调试建议

1. Run `pytest` first.
2. Run `doctor` second.
3. Run a small `search` third.
4. Inspect `outputs/demo` last.
