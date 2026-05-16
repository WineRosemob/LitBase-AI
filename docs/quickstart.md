# Quickstart

## Goal | 目标

Get a fresh clone into a runnable state with a minimal diagnostic and demo search.

让一个全新 clone 的仓库尽快进入可运行状态，并完成最小诊断与示例检索。

## Fast Path | 最短路径

```bash
bash scripts/setup_conda_env.sh
cp examples/example.env .env
litbase-ai doctor --env-file .env --output-dir outputs/doctor
litbase-ai search --env-file .env --topic "climate change integrated assessment model" --limit 20 --year-from 2018 --disable-cnki --output-dir outputs/demo
```

## Required edits | 需要手动填写

- `OPENALEX_MAILTO`
- `UNPAYWALL_EMAIL`
- `LLM_API_KEY`

## Notes | 说明

- `examples/example.env` is a safe public placeholder file.
- `examples/example.env` 是安全的公开占位符模板。
- `doctor` may report external-service issues without indicating a code bug.
- `doctor` 可能提示外部服务问题，这不一定代表代码有错。
- Install `python -m pip install '.[embeddings]'` only if you plan to enable embedding scoring.
- 只有在你准备启用 embedding score 时，才需要额外安装 `python -m pip install '.[embeddings]'`。
- The repository ignores `outputs/`, PDFs, cookies, logs, and local env files by default.
- 仓库默认忽略 `outputs/`、PDF、cookie、日志和本地 env 文件。
