# LitBase-AI

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/<your-username>/LitBase-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/LitBase-AI/actions/workflows/ci.yml)

LitBase-AI is a bilingual research workflow toolkit for multi-source literature retrieval, scoring, compliant PDF access, and structured export.

LitBase-AI 是一个面向研究工作流的双语文献工具，用于多源检索、评分、合规下载与结构化导出。

## Features | 功能概览

- Multi-source literature search: OpenAlex, Crossref, arXiv, Semantic Scholar, optional CNKI.
- 多源文献检索：OpenAlex、Crossref、arXiv、Semantic Scholar，以及可选 CNKI。
- Query expansion for broad recall and better topic coverage.
- 查询扩展，提升召回范围与主题覆盖。
- Rule-based scoring, evidence extraction, and optional embedding score.
- 规则评分、证据提取，以及可选 embedding score。
- LLM rubric scoring for higher-quality paper selection.
- 使用 LLM rubric 对候选文献做更细粒度评分。
- Literature cards plus BibTeX / Excel / JSONL / Markdown outputs.
- 生成文献卡片，并导出 BibTeX / Excel / JSONL / Markdown。
- Open-access PDF download and user-authorized institutional access flows.
- 支持开放获取 PDF 下载与用户已授权机构访问流程。
- `doctor` command for environment and source diagnostics.
- 提供 `doctor` 诊断命令，用于检查环境与数据源状态。

## Workflow | 工作流程

```mermaid
flowchart LR
    A[Topic] --> B[Query Expansion]
    B --> C[Multi-source Search]
    C --> D[Merge and Deduplicate]
    D --> E[Rule Score]
    E --> F[LLM Rubric Score]
    F --> G[Select Papers]
    G --> H[Download OA PDFs]
    H --> I[Export Results]
```

Text form:

`Topic -> Query Expansion -> Multi-source Search -> Merge & Deduplicate -> Rule Score -> LLM Score -> Select Papers -> Download OA PDFs -> Export Results`

## Installation | 安装方式

### Option 1: Conda one-click setup | Conda 一键安装

```bash
bash scripts/setup_conda_env.sh
bash scripts/setup_conda_env.sh my-env-name
```

### Option 2: Manual pip setup | 手动 pip 安装

```bash
conda create -n litbase-ai python=3.11 -y
conda run -n litbase-ai python -m pip install -r requirements.txt
conda run -n litbase-ai python -m pip install -e .
conda run -n litbase-ai python -m playwright install firefox

# Optional: install embedding support
conda run -n litbase-ai python -m pip install '.[embeddings]'
```

## Quickstart | 快速开始

1. Copy the example env file.
1. 复制示例配置文件。

```bash
cp examples/example.env .env
```

2. Fill in the placeholders.
2. 填写占位符配置。

Required / 常见必填项:

- `OPENALEX_MAILTO=your_email@example.com`
- `UNPAYWALL_EMAIL=your_email@example.com`
- `LLM_API_KEY=your_llm_api_key_here`

3. Run `doctor` first.
3. 先运行 `doctor` 诊断。

```bash
litbase-ai doctor --env-file .env --output-dir outputs/doctor
```

4. Run a small search demo.
4. 运行一个小规模检索示例。

```bash
litbase-ai search \
  --env-file .env \
  --topic "climate change integrated assessment model" \
  --limit 20 \
  --year-from 2018 \
  --disable-cnki \
  --progress-style rich \
  --output-dir outputs/demo
```

## Configuration | 配置说明

Primary public template:

- [`examples/example.env`](examples/example.env)
- [`.env.example`](.env.example)

Important variables:

- `OPENALEX_MAILTO`: contact email for OpenAlex requests.
- `UNPAYWALL_EMAIL`: email required by Unpaywall.
- `LLM_API_KEY`: API key for LLM scoring/query expansion.
- `LLM_BASE_URL` / `LLM_MODEL`: LLM endpoint and model.
- `SEMANTIC_SCHOLAR_API_KEY`: optional Semantic Scholar key.
- `sentence-transformers`: optional extra for embedding score workflows.
- `ENABLE_SCIHUB=false`: legacy source, disabled by default.
- `ENABLE_LIBGEN=false`: legacy source, disabled by default.
- `ENABLE_ARXIV_DOWNLOAD=true`: legal arXiv PDF support.
- `ENABLE_EZPROXY=false`: optional authorized institutional access.
- `ENABLE_INST_PROXY=false`: optional WebVPN / institutional proxy access.

Config priority:

- `--env-file`
- `.env`
- `.env.deepseek` (legacy compatibility)
- system environment variables

More details: [`docs/configuration.md`](docs/configuration.md)

## CLI | 命令行使用

```bash
litbase-ai --help
litbase-ai doctor --env-file examples/example.env --output-dir outputs/doctor
litbase-ai init --env-file examples/example.env --non-interactive
litbase-ai search --env-file examples/example.env --topic "energy transition" --limit 20 --year-from 2020 --output-dir outputs/demo
```

Main commands:

- `doctor`: runtime and data-source diagnostics.
- `init`: cookie/bootstrap policy for authorized institutional access.
- `search`: full retrieval, scoring, export, and optional download pipeline.

## Outputs | 输出目录说明

Typical output layout:

```text
outputs/demo/
├── expanded_query.json
├── papers_raw.jsonl
├── papers_scored.xlsx
├── papers_selected.xlsx
├── papers_cards.xlsx
├── references.bib
├── search_diagnostics.json
├── summary_report.md
├── run.log
├── pdf/
└── literature_cards/
```

Output and cache folders are intentionally ignored by Git.

## Project Structure | 项目结构

```text
LitBase-AI/
├── README.md
├── LICENSE.md
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── docs/
├── examples/
├── scripts/
├── tests/
└── litbase_ai/
    ├── search/
    ├── query/
    ├── scoring/
    ├── download/
    ├── enrich/
    ├── storage/
    ├── utils/
    ├── config.py
    ├── pipeline.py
    ├── cli.py
    └── models.py
```

AI agent onboarding guide:

- [`docs/ai_skill.md`](docs/ai_skill.md)

## Compliance and Copyright | 合规与版权声明

- This project is intended for lawful scholarly metadata retrieval, open-access PDF download, and user-authorized institutional access only.
- 本项目仅用于合法、合规的学术元数据检索、开放获取文献下载，以及用户已获授权的机构访问。
- Do not use this project to bypass paywalls, licenses, or publisher access controls.
- 不要使用本项目绕过付费墙、许可协议或出版商访问控制。
- `Sci-Hub` and `LibGen` support remains in the codebase only as legacy / experimental / disabled-by-default functionality.
- `Sci-Hub` 与 `LibGen` 相关支持仅作为遗留/实验能力保留在代码中，默认关闭，不作为推荐方案。
- Preferred legal sources include OpenAlex, Crossref, Semantic Scholar, arXiv, Unpaywall, PMC, and properly authorized institutional proxy flows.
- 更推荐使用 OpenAlex、Crossref、Semantic Scholar、arXiv、Unpaywall、PMC 以及合规配置的机构代理访问流程。

## FAQ | 常见问题

### Why are Sci-Hub and LibGen disabled by default?

Because this repository is prepared for public release and compliant usage. They remain opt-in legacy code paths, not the recommended workflow.

因为本仓库按公开发布与合规使用整理，所以这两类能力保留但不默认开启，也不作为推荐工作流。

### Why did `doctor` report external failures?

`doctor` distinguishes code/runtime readiness from external availability. Missing API keys, rate limits, blocked network access, or absent browser binaries can appear in the report without meaning the code is broken.

`doctor` 会区分“代码可运行”和“外部服务可用性”。缺失 API key、限流、网络受限或浏览器二进制缺失，不等于代码错误。

### How should other developers or AI agents start?

Read [`docs/quickstart.md`](docs/quickstart.md) first, then follow [`docs/ai_skill.md`](docs/ai_skill.md).

先读 [`docs/quickstart.md`](docs/quickstart.md)，再看 [`docs/ai_skill.md`](docs/ai_skill.md)。
