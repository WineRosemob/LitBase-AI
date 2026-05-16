# Configuration

## Recommended files | 推荐文件

- `examples/example.env`: public safe example.
- `.env.example`: fuller template for local setups.
- `.env.deepseek.example`: legacy naming template for users who want a DeepSeek-specific env filename.

## Core variables | 核心变量

- `OPENALEX_MAILTO`: email used for OpenAlex requests.
- `UNPAYWALL_EMAIL`: email required by Unpaywall.
- `LLM_API_KEY`: primary API key for LLM-based scoring and query expansion.
- `LLM_BASE_URL`: LLM endpoint base URL.
- `LLM_MODEL`: model name.
- `SEMANTIC_SCHOLAR_API_KEY`: optional source key.
- `sentence-transformers`: optional extra dependency used only when embedding scoring is enabled.

## Download policy switches | 下载策略开关

- `ENABLE_SCIHUB=false`
- `ENABLE_LIBGEN=false`
- `ENABLE_ARXIV_DOWNLOAD=true`
- `ENABLE_EZPROXY=false`
- `ENABLE_INST_PROXY=false`

Public-release default means:

- legacy/non-authorized sources are off by default;
- legal OA and user-authorized institutional flows are prioritized.

公开版默认表示：

- 遗留/非授权来源默认关闭；
- 优先使用合法开放获取与用户已授权的机构访问流程。

## Institutional access | 机构访问相关

Optional variables:

- `EZPROXY_TEMPLATE`
- `EZPROXY_COOKIE_FILE`
- `INST_PROXY_MODE`
- `INST_PROXY_URL`
- `INST_PROXY_COOKIE_FILE`
- `INST_PROXY_SCHOOL`
- `WEBVPN_URL`
- `WEBVPN_USERNAME`
- `WEBVPN_PASSWORD`
- `WEBVPN_AUTO_LOGIN`

Important:

- Do not commit real credentials.
- 不要提交真实账号密码。
- Do not enable institutional flows unless you are authorized to use them.
- 仅在你确有授权时启用机构访问流程。
- `INST_PROXY_SCHOOL` can map into the bundled school database, but no school is used as a public default example.
- `INST_PROXY_SCHOOL` 可以映射到内置学校库，但公开文档不再以任何具体学校作为默认示例。
