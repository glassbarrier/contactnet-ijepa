# AGENTS.md — AI 代理快速上手说明

目的
--
本文件为代码库提供给 AI 编程代理的一组简明指引，帮助代理快速定位构建/测试命令、理解项目约定并安全地提出或生成更改。

使用场景（简要）
--
- 当代理需要运行构建或测试时，先在仓库中查找常见构建清单（`package.json`, `pyproject.toml`, `Makefile`, `build.gradle`, `pom.xml`, 等）。
- 当代理需要变更代码，优先搜索并链接现有文档而不是复制其内容（见“Link, don't embed”原则）。

快速检查清单
--
- 构建/测试命令：搜索并读取 `package.json`（`scripts`）、`pyproject.toml`、`Makefile`、`setup.py`、`gradle` 文件。把找到的命令作为建议的运行步骤。
- 关键文档：查找 `README.md`、`CONTRIBUTING.md`、`ARCHITECTURE.md` 并在建议中以链接引用。

约定（最小化且高效）
--
- 提交/PR：遵循常见的语义化提交格式（例如 `feat:`, `fix:`, `docs:`），并在 PR 描述中包含变更动机与如何验证的步骤。
- 格式化：优先使用仓库已有的格式化工具（例如 `prettier`, `black`, `gofmt`）；若未发现，建议不要自动大面积 reformat，先询问维护者。
- 测试：修改功能代码时同时添加或更新单元测试，并在 PR 描述中说明如何运行这些测试。

安全与边界
--
- 不要在代理生成的更改中硬编码凭据、密钥或任何敏感信息。提醒并引导开发者使用 secret 管理方案。
- 对大型或破坏性更改（数据库迁移、删除表、批量数据处理等），在 PR 中强烈建议写明回滚策略并标记为需要人工审核。

关键文件/目录（代理首要检查）
--
- 根目录下的 `README.md`（项目概况）
- 构建文件：`package.json`, `pyproject.toml`, `Makefile`, `pom.xml`, `build.gradle`
- 测试目录（常见为 `tests/`, `spec/`）
- 配置/脚本目录（例如 `scripts/`, `.github/workflows/`）

建议的后续自定义
--
- 创建 `.github/copilot-instructions.md` 来放置与仓库密切相关的代理行为示例和禁忌。
- 为常用任务创建小型 skill（例如：`run-tests`, `format-code`, `create-pr-template`）。

如何反馈与迭代
--
如果你希望我把此文件转为 `.github/copilot-instructions.md` 或为特定子系统（frontend/backend）创建更详细的说明，告诉我目标区域和任何现有文档或约定，我会进行下一步迭代。

— 结束 —
