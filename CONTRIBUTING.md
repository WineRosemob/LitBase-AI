# Contributing to LitBase-AI

感谢你的贡献兴趣！/ Thank you for your interest in contributing!

## 开发环境搭建 / Development Setup

### 1. 克隆仓库 / Clone

```bash
git clone https://github.com/<your-username>/LitBase-AI.git
cd LitBase-AI
```

### 2. 创建虚拟环境 / Create Environment

**Option A — Conda (推荐 / recommended):**

```bash
conda env create -f environment.yml
conda activate litbase-ai
```

**Option B — venv + pip:**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[embeddings]"
```

### 3. 安装 Pre-commit 钩子 / Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### 4. 安装 Playwright 浏览器 / Install Playwright Browsers

```bash
playwright install --with-deps chromium
```

## 代码风格 / Code Style

- **格式化 / Formatting**: [Ruff](https://docs.astral.sh/ruff/formatter/)
- **Lint**: [Ruff](https://docs.astral.sh/ruff/linter/)
- **类型提示 / Type Hints**: 鼓励使用，但非强制 / Encouraged but not required
- **提交信息 / Commit Messages**: 使用中文或英文均可

提交前 pre-commit 会自动运行 Ruff 格式化和检查。

## 运行测试 / Running Tests

```bash
# 运行全部测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_config.py -v

# 带覆盖率
pip install pytest-cov
pytest tests/ -v --cov=litbase_ai --cov-report=html
```

## 项目结构 / Project Structure

```
LitBase-AI/
├── litbase_ai/          # 核心包 / Core package
│   ├── cli.py           # CLI 入口 / Entry point
│   ├── config.py        # 配置管理 / Configuration
│   ├── pipeline.py      # 流水线编排 / Pipeline
│   ├── search/          # 文献检索 / Literature search
│   ├── download/        # PDF 下载 / PDF download
│   ├── scoring/         # 相关性评分 / Relevance scoring
│   ├── enrich/          # 元数据增强 / Metadata enrichment
│   ├── query/           # 查询扩展 / Query expansion
│   ├── storage/         # 结果存储 / Result storage
│   ├── prompts/         # LLM 提示词 / LLM prompts
│   └── utils/           # 工具函数 / Utilities
├── tests/               # 测试 / Tests
├── docs/                # 文档 / Documentation
├── examples/            # 示例 / Examples
└── scripts/             # 脚本 / Scripts
```

## Pull Request 流程 / PR Process

1. **Fork 仓库** / Fork the repository
2. **创建分支** / Create a feature branch: `git checkout -b feature/your-feature`
3. **编写代码和测试** / Write code and tests
4. **运行测试** / Run `pytest tests/ -v` 确保全部通过
5. **确保 pre-commit 通过** / Ensure pre-commit checks pass
6. **提交并推送** / Commit and push
7. **创建 Pull Request** / Open a PR against `main`

### PR 检查清单 / PR Checklist

- [ ] 测试全部通过 / All tests pass
- [ ] Pre-commit 钩子通过 / Pre-commit hooks pass
- [ ] 新功能有测试覆盖 / New features have test coverage
- [ ] 相关文档已更新 / Relevant docs updated
- [ ] 未包含敏感信息（API key 等）/ No sensitive info committed

## 注意事项 / Important Notes

- **绝对不要** 提交任何 API key、密码或凭证文件
- **Never** commit `.env` files, cookies, or `storage_state.json`
- 修改 `.env.example` 时，确保不包含真实密钥
- When updating `.env.example`, ensure no real keys are included

## 问题反馈 / Questions?

请在 [GitHub Issues](https://github.com/<your-username>/LitBase-AI/issues) 中提出。
