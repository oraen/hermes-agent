# Hermes Agent 源码架构详解

## 项目概述

Hermes Agent 是一个功能丰富的 AI 代理系统，支持 CLI 交互、消息平台网关、编辑器集成（ACP）、定时任务、批量处理、RL 训练等多种运行模式。项目以 Python 为主语言，辅以 Nix 打包、Node.js 前端和 Docusaurus 文档站。

---

## 一、根目录核心模块

根目录包含系统入口和全局共享基础设施，被所有子目录引用：

| 文件 | 职责 |
|------|------|
| `hermes_constants.py` | **全局常量与路径工具**。提供 `get_hermes_home()`、`display_hermes_home()`、`OPENROUTER_BASE_URL` 等，是依赖链最底层的模块，无项目内部依赖，可被任何模块安全导入 |
| `hermes_logging.py` | **集中式日志配置**。`setup_logging()` 提供日志文件轮转、密钥脱敏、会话上下文标记，CLI 和 Gateway 启动时调用 |
| `hermes_time.py` | **时区感知时钟**。基于 `hermes_constants.get_config_path()` 读取用户时区配置，提供统一的 `now()` 函数 |
| `hermes_state.py` | **SQLite 会话存储**。使用 FTS5 全文搜索，WAL 模式支持并发读写，存储会话元数据、消息历史和模型配置 |
| `utils.py` | **通用工具函数**。`is_truthy_value()`、`env_var_enabled()`、`atomic_yaml_write()`、`atomic_json_write()` 等，被 agent/、tools/、gateway/ 广泛引用 |
| `model_tools.py` | **工具编排层**。触发 `tools/registry.py` 的自动发现机制，提供 `get_tool_definitions()` 和 `handle_function_call()` 公共 API，是 run_agent.py 和 CLI 的桥梁 |
| `toolsets.py` | **工具集定义**。定义 `_HERMES_CORE_TOOLS` 列表和工具集组合/解析逻辑，`model_tools.py` 和 `batch_runner.py` 的直接依赖 |
| `toolset_distributions.py` | **工具集概率分布**。为批量数据生成定义工具集的选择概率，依赖 `toolsets.py`，被 `batch_runner.py` 引用 |
| `run_agent.py` | **AIAgent 核心**。约 11600 行，包含完整的代理对话循环：消息管理、工具调用、上下文压缩触发、失败转移，依赖 agent/ 和 model_tools |
| `cli.py` | **HermesCLI 交互式终端**。约 10300 行，基于 prompt_toolkit 和 Rich，处理斜杠命令、皮肤主题、自动补全，依赖 agent/、hermes_cli/、model_tools、tools/ |
| `batch_runner.py` | **并行批量处理**。使用多进程运行 AIAgent 处理 JSONL 数据集，支持断点续跑，依赖 run_agent、model_tools、toolset_distributions |
| `trajectory_compressor.py` | **轨迹压缩**。后处理已完成的代理轨迹，在保留训练信号质量的前提下压缩到目标 token 预算，依赖 agent/retry_utils |
| `mini_swe_runner.py` | **SWE 任务运行器**。使用 Hermes 的 Docker/Modal/Local 环境执行 SWE-bench 任务，输出 Hermes 格式轨迹 |
| `rl_cli.py` | **RL 训练 CLI**。专用 CLI 运行器，连接 tinker-atropos 子模块进行强化学习训练 |
| `mcp_serve.py` | **MCP 服务器**。以 stdio 模式暴露消息对话为 MCP 工具，供 Claude Code/Cursor 等外部客户端调用 |
| `hermes` | **Shell 入口脚本**。`/usr/bin/hermes` 的命令行入口点 |

---

## 二、各目录详解

### 1. `agent/` — 代理内部逻辑（33 个文件）

**文件类型：** 纯 Python 模块，无子目录

**职责：** 实现 AI 代理的核心智能逻辑，包括提示构建、上下文管理、模型适配、凭证池、错误分类等

| 关键文件 | 功能 |
|---------|------|
| `prompt_builder.py` (46KB) | 系统提示组装：注入技能、记忆、目录提示、SOUL.md 等 |
| `auxiliary_client.py` (115KB) | 辅助 LLM 客户端：视觉分析、摘要生成、异步调用 |
| `context_compressor.py` (53KB) | 自动上下文压缩：当 token 超限时触发中间轮次压缩 |
| `context_engine.py` | 上下文引擎接口 |
| `context_references.py` | 上下文引用管理 |
| `credential_pool.py` (59KB) | 多凭证轮转池：支持多 API key 轮询、故障转移 |
| `model_metadata.py` (45KB) | 模型元数据：上下文长度、token 估算、模型能力映射 |
| `models_dev.py` (20KB) | models.dev 注册表集成：提供商感知的模型上下文信息 |
| `prompt_caching.py` | Anthropic 提示缓存逻辑 |
| `error_classifier.py` (29KB) | API 错误分类：区分限流、认证、超时等，指导重试策略 |
| `retry_utils.py` | 抖动退避重试工具 |
| `rate_limit_tracker.py` | 速率限制追踪 |
| `nous_rate_guard.py` | Nous 专用速率守卫 |
| `smart_model_routing.py` | 智能模型路由 |
| `memory_manager.py` | 记忆管理：构建记忆上下文块 |
| `memory_provider.py` | 记忆提供者抽象 |
| `display.py` (39KB) | KawaiiSpinner 动画、工具预览格式化、皮肤渲染 |
| `skill_commands.py` | 技能斜杠命令：扫描 `~/.hermes/skills/` 并注入为用户消息 |
| `skill_utils.py` | 技能工具函数 |
| `anthropic_adapter.py` (63KB) | Anthropic API 适配器 |
| `bedrock_adapter.py` (43KB) | AWS Bedrock 适配器 |
| `gemini_cloudcode_adapter.py` (28KB) | Gemini Cloud Code 适配器 |
| `google_code_assist.py` | Google Code Assist 集成 |
| `google_oauth.py` (38KB) | Google OAuth 认证流程 |
| `copilot_acp_client.py` | Copilot ACP 客户端 |
| `redact.py` | 敏感信息脱敏（密钥、令牌） |
| `usage_pricing.py` (25KB) | 用量和定价计算 |
| `insights.py` (33KB) | 代理洞察与遥测 |
| `title_generator.py` | 对话标题生成 |
| `trajectory.py` | 轨迹保存辅助 |
| `subdirectory_hints.py` | 子目录提示追踪器 |
| `manual_compression_feedback.py` | 手动压缩反馈 |

**依赖方向：** 
- ← 引用：`hermes_constants`、`utils`、`hermes_cli.auth`、`hermes_cli.config`
- ← 被引用：`run_agent.py`、`cli.py`、`tools/`（部分工具）、`hermes_cli/`（部分子命令）

---

### 2. `tools/` — 工具实现（56 个文件 + 3 子目录）

**文件类型：** Python 工具模块 + 子目录（environments/、browser_providers/、neutts_samples/）

**职责：** 每个文件实现一个独立工具，通过 `registry.register()` 自注册。所有工具处理器必须返回 JSON 字符串

| 关键文件 | 功能 |
|---------|------|
| `registry.py` (19KB) | **工具注册中心**。提供 `register()`、`discover_builtin_tools()`、`dispatch()`，无项目内依赖，被所有工具文件和 model_tools 引用 |
| `terminal_tool.py` (75KB) | 终端编排：本地/Docker/SSH/Modal 执行环境 |
| `process_registry.py` (49KB) | 后台进程管理 |
| `browser_tool.py` (95KB) | 浏览器自动化（Browserbase/CamoFox） |
| `web_tools.py` (87KB) | Web 搜索/提取（Parallel + Firecrawl） |
| `file_tools.py` (37KB) | 文件读/写/搜索/补丁 |
| `file_operations.py` (48KB) | 文件操作扩展 |
| `mcp_tool.py` (102KB) | MCP 客户端实现 |
| `delegate_tool.py` (48KB) | 子代理委派 |
| `code_execution_tool.py` (55KB) | 代码执行沙箱 |
| `send_message_tool.py` (56KB) | 消息发送（跨平台） |
| `skills_tool.py` (52KB) | 技能执行/管理 |
| `skills_hub.py` (112KB) | 技能市场搜索/安装 |
| `tts_tool.py` (51KB) | 文本转语音 |
| `voice_mode.py` (39KB) | 语音模式 |
| `vision_tools.py` (31KB) | 视觉分析 |
| `image_generation_tool.py` (31KB) | 图像生成 |
| `approval.py` (39KB) | 危险命令检测 |
| `memory_tool.py` (23KB) | 记忆工具 |
| `todo_tool.py` (10KB) | 待办工具（代理级别拦截） |
| `rl_training_tool.py` (57KB) | RL 训练工具 |
| `cronjob_tools.py` (21KB) | Cron 任务工具 |
| `transcription_tools.py` (27KB) | 语音转录 |
| `homeassistant_tool.py` (18KB) | 智能家居控制 |
| `session_search_tool.py` (23KB) | 会话搜索 |
| `skill_manager_tool.py` (29KB) | 技能管理器 |
| `mixture_of_agents_tool.py` (22KB) | 多代理混合 |
| `checkpoint_manager.py` (25KB) | 检查点管理 |
| `patch_parser.py` (21KB) | 补丁解析 |
| `tirith_security.py` (26KB) | 安全策略执行 |
| `skills_guard.py` (36KB) | 技能安全守卫 |
| `skills_sync.py` (15KB) | 技能同步 |
| `fuzzy_match.py` (21KB) | 模糊匹配 |
| `url_safety.py` | URL 安全检查 |
| `website_policy.py` | 网站策略 |
| `clarify_tool.py` | 澄清工具 |
| `interrupt.py` | 中断信号 |
| `credential_files.py` | 凭证文件管理 |
| `budget_config.py` | 预算配置 |
| `tool_result_storage.py` | 工具结果存储 |
| `debug_helpers.py` | 调试辅助 |
| `env_passthrough.py` | 环境变量透传 |
| `mcp_oauth.py` / `mcp_oauth_manager.py` | MCP OAuth 管理 |
| `openrouter_client.py` | OpenRouter 客户端 |
| `xai_http.py` | xAI HTTP 客户端 |
| `ansi_strip.py` | ANSI 转义序列清理 |
| `binary_extensions.py` | 二进制扩展名列表 |
| `browser_camofox.py` / `browser_camofox_state.py` | CamoFox 浏览器 |
| `neutts_synth.py` | NeuTTS 语音合成 |
| `managed_tool_gateway.py` | 托管工具网关 |
| `tool_backend_helpers.py` | 工具后端辅助 |

**子目录：**

- `tools/environments/` — 终端后端实现（local、docker、ssh、modal、daytona、singularity）
- `tools/browser_providers/` — 浏览器提供者（browserbase、browser_use、firecrawl）
- `tools/neutts_samples/` — NeuTTS 语音样本

**依赖方向：**
- ← 引用：`hermes_constants`（大量）、`agent.auxiliary_client`（视觉/LLM 调用）、`agent.redact`（脱敏）、`hermes_cli.config`、`toolsets`
- ← 被引用：`model_tools.py`（发现和调度）、`run_agent.py`（直接引用部分工具）、`cli.py`（回调设置）、`environments/`（RL 环境）

---

### 3. `hermes_cli/` — CLI 子命令与配置（53 个文件）

**文件类型：** Python 模块，无子目录

**职责：** 实现 `hermes` 命令的所有子命令、配置管理、认证、皮肤主题等

| 关键文件 | 功能 |
|---------|------|
| `main.py` (267KB) | **入口点**。所有 `hermes` 子命令的 argparse 定义和分发 |
| `config.py` (144KB) | `DEFAULT_CONFIG`、`OPTIONAL_ENV_VARS`、配置迁移、`load_config()` |
| `auth.py` (133KB) | 提供商凭证解析：API key 查找、OAuth 刷新、Bedrock 凭证链 |
| `gateway.py` (135KB) | Gateway 生命周期管理：start/stop/status/install/uninstall |
| `setup.py` (129KB) | 交互式设置向导 |
| `web_server.py` (90KB) | 内置 Web 服务器 |
| `tools_config.py` (77KB) | `hermes tools` — 工具启用/禁用（使用 curses 而非 simple_term_menu） |
| `models.py` (80KB) | 模型目录、提供商模型列表 |
| `doctor.py` (52KB) | 配置和依赖检查 |
| `commands.py` (50KB) | `COMMAND_REGISTRY` — 斜杠命令中心定义（CommandDef 列表） |
| `skills_hub.py` (49KB) | `/skills` 斜杠命令实现 |
| `model_switch.py` (43KB) | `/model` 切换流水线（CLI + Gateway 共用） |
| `profiles.py` (38KB) | 多实例配置文件管理 |
| `skin_engine.py` (40KB) | 皮肤/主题引擎：SkinConfig 数据类、内置皮肤、YAML 加载 |
| `plugins.py` (31KB) | 插件系统 |
| `plugins_cmd.py` (39KB) | `/plugins` 斜杠命令 |
| `runtime_provider.py` (42KB) | 运行时提供商管理 |
| `auth_commands.py` (24KB) | `hermes auth` 子命令 |
| `banner.py` (22KB) | 启动横幅渲染 |
| `backup.py` (22KB) | 配置备份 |
| `status.py` (20KB) | 状态显示 |
| `claw.py` (27KB) | OpenClaw 迁移工具 |
| `mcp_config.py` (27KB) | MCP 配置管理 |
| `nous_subscription.py` (28KB) | Nous 订阅管理 |
| `tips.py` (27KB) | 提示/技巧系统 |
| `memory_setup.py` (16KB) | Honcho 记忆集成设置 |
| `curses_ui.py` (17KB) | curses 交互式 UI 组件 |
| `logs.py` (13KB) | 日志查看 |
| `completion.py` (11KB) | Shell 自动补全生成 |
| `copilot_auth.py` (10KB) | GitHub Copilot 认证 |
| `dingtalk_auth.py` (10KB) | 钉钉认证 |
| `cron.py` (11KB) | `hermes cron` 子命令 |
| `callbacks.py` (8KB) | 终端回调：澄清、sudo、审批 |
| `skills_config.py` (7KB) | 技能配置 |
| `clipboard.py` (15KB) | 剪贴板操作 |
| `debug.py` (16KB) | 调试工具 |
| `dump.py` (11KB) | 会话数据导出 |
| `env_loader.py` (5KB) | `.env` 文件加载器 |
| `default_soul.py` | 默认 SOUL.md 内容 |
| `colors.py` | 颜色常量 |
| `cli_output.py` | CLI 输出格式化 |
| `codex_models.py` | Codex 模型映射 |
| `model_normalize.py` | 模型名规范化 |
| `pairing.py` | 设备配对 |
| `platforms.py` | 平台常量 |
| `uninstall.py` | 卸载逻辑 |
| `webhook.py` | Webhook 管理 |

**依赖方向：**
- ← 引用：`hermes_constants`（大量）、`agent.credential_pool`、`agent.models_dev`、`tools.tool_backend_helpers`、`tools.managed_tool_gateway`、`gateway.status`、`gateway.restart`
- ← 被引用：`run_agent.py`、`cli.py`、`gateway/run.py`、`cron/scheduler.py`、`trajectory_compressor.py`

---

### 4. `gateway/` — 消息平台网关（15 个文件 + 2 子目录）

**文件类型：** Python 模块 + 子目录（platforms/、builtin_hooks/）

**职责：** 实现 Hermes 作为消息平台机器人的运行模式，管理会话、消息分发、跨平台投递

| 关键文件 | 功能 |
|---------|------|
| `run.py` (466KB) | **主循环**。GatewayRunner 类：平台启动、斜杠命令分发、代理实例缓存、SSL 证书检测 |
| `session.py` (42KB) | SessionStore — 对话持久化 |
| `config.py` (57KB) | 网关配置加载与验证 |
| `stream_consumer.py` (35KB) | 流式响应消费 |
| `status.py` (16KB) | 网关运行状态、PID 管理、作用域锁 |
| `delivery.py` (9KB) | DeliveryRouter — 跨平台消息投递 |
| `channel_directory.py` (10KB) | 频道目录管理 |
| `pairing.py` (11KB) | 设备配对 |
| `hooks.py` (6KB) | 钩子框架 |
| `display_config.py` (7KB) | 显示配置 |
| `mirror.py` (4KB) | 消息镜像 |
| `session_context.py` (5KB) | 会话上下文 |
| `sticker_cache.py` (3KB) | 贴纸缓存 |
| `restart.py` (1KB) | 网关重启 |

**子目录：**

- `gateway/platforms/` — **25 个平台适配器**（telegram、discord、slack、whatsapp、signal、matrix、feishu、wecom、weixin、qqbot、dingtalk、homeassistant、email、sms、webhook、bluebubbles、api_server 等），每个适配器继承 `base.py` 的 PlatformAdapter 基类
- `gateway/builtin_hooks/` — 内置钩子（boot_md 等）

**依赖方向：**
- ← 引用：`hermes_constants`、`hermes_cli.config`、`hermes_cli.env_loader`、`utils`、`tools.url_safety`
- ← 被引用：`hermes_cli/gateway.py`（生命周期管理）、`hermes_cli/web_server.py`（状态查询）、`mcp_serve.py`（消息桥接）

---

### 5. `acp_adapter/` — ACP 服务器（9 个文件）

**文件类型：** Python 模块，无子目录

**职责：** 实现 Agent Client Protocol 服务器，让 VS Code / Zed / JetBrains 等编辑器通过 ACP 协议与 Hermes 交互

| 文件 | 功能 |
|------|------|
| `server.py` (28KB) | ACP 服务器主逻辑：会话管理、提示处理、命令分发 |
| `session.py` (18KB) | ACP 会话管理 |
| `tools.py` (7KB) | ACP 工具暴露 |
| `events.py` (6KB) | ACP 事件处理 |
| `permissions.py` (3KB) | 权限控制 |
| `auth.py` (1KB) | ACP 认证 |
| `entry.py` (3KB) | 入口点 |
| `__main__.py` | `python -m acp_adapter` 入口 |
| `__init__.py` | 包初始化 |

**依赖方向：**
- ← 引用：`hermes_constants`（路径工具）、`acp`（第三方 ACP 协议库）
- ← 被引用：`hermes_cli/main.py`（`hermes acp` 子命令启动）

---

### 6. `cron/` — 定时任务（3 个文件）

**文件类型：** Python 模块，无子目录

**职责：** Cron 作业调度器，由 Gateway 后台线程定期调用

| 文件 | 功能 |
|------|------|
| `scheduler.py` (43KB) | 调度引擎：tick() 检查到期任务并执行，文件锁防止重复执行 |
| `jobs.py` (27KB) | 作业定义和持久化 |
| `__init__.py` | 包初始化，导出 `get_job` |

**依赖方向：**
- ← 引用：`hermes_constants`、`hermes_cli.config`、`hermes_time`
- ← 被引用：`cli.py`（`/cron` 命令）、`gateway/run.py`（定期 tick）

---

### 7. `environments/` — RL 训练环境（7 个文件 + 4 子目录）

**文件类型：** Python 模块 + 子目录

**职责：** 为强化学习训练提供 Atropos 兼容的 Gym 环境

| 关键文件 | 功能 |
|---------|------|
| `hermes_base_env.py` (29KB) | 基础环境：工具定义、token 预算 |
| `agentic_opd_env.py` (51KB) | OPD 环境 |
| `web_research_env.py` (29KB) | Web 研究环境 |
| `agent_loop.py` (24KB) | 代理循环封装 |
| `tool_context.py` (17KB) | 工具上下文 |
| `patches.py` | 补丁工具 |

**子目录：**
- `environments/benchmarks/` — 基准测试环境（terminalbench 等）
- `environments/hermes_swe_env/` — SWE-bench 环境
- `environments/terminal_test_env/` — 终端测试环境
- `environments/tool_call_parsers/` — 工具调用解析器（12 个文件）

**依赖方向：**
- ← 引用：`model_tools`、`tools.terminal_tool`、`tools.browser_tool`、`tools.budget_config`
- ← 被引用：`rl_cli.py`、`tinker-atropos/`

---

### 8. `plugins/` — 插件系统（3 子目录）

**文件类型：** Python 包 + 子目录

**职责：** 可插拔的扩展系统

| 子目录 | 功能 |
|--------|------|
| `plugins/memory/` | 记忆后端（8 个实现：honcho、mem0、supermemory、byterover、hindsight、holographic、openviking、retaindb） |
| `plugins/context_engine/` | 上下文引擎插件 |
| `plugins/example-dashboard/` | 示例仪表盘插件 |

**依赖方向：**
- ← 引用：`hermes_constants`
- ← 被引用：`agent/memory_provider.py`、`hermes_cli/plugins.py`

---

### 9. `skills/` — 技能定义（26 子目录）

**文件类型：** YAML/Markdown/Python 技能定义文件，按领域组织

**职责：** 可安装的技能包，每个技能包含 prompt 模板和可选的工具定义

| 领域目录 | 示例技能 |
|----------|---------|
| `apple/` | Apple Script 控制 |
| `autonomous-ai-agents/` | 自主代理 |
| `creative/` | 创意生成（9 个子技能） |
| `data-science/` | 数据科学 |
| `devops/` | DevOps/Webhook |
| `github/` | GitHub 集成（6 个子技能） |
| `mcp/` | MCP 集成 |
| `mlops/` | MLops（9 个子技能） |
| `productivity/` | 生产力工具 |
| `research/` | 研究辅助 |
| `software-development/` | 软件开发（6 个子技能） |
| 其他 | diagramming、email、feeds、gaming、gifs、media、smart-home、social-media 等 |

**依赖方向：**
- ← 无代码依赖（纯数据文件）
- ← 被引用：`agent/skill_commands.py`（扫描加载）、`tools/skills_tool.py`（执行）、`tools/skills_hub.py`（市场搜索安装）

---

### 10. `optional-skills/` — 可选技能源（13 子目录）

**文件类型：** 与 `skills/` 相同格式的技能定义

**职责：** 社区/第三方技能的额外源，结构与 skills/ 平行

---

### 11. `tests/` — 测试套件（约 3000 个测试，12 子目录）

**文件类型：** Python 测试文件（pytest）

| 子目录 | 测试内容 |
|--------|---------|
| `tests/agent/` (41 文件) | agent/ 模块测试 |
| `tests/tools/` (135 文件) | tools/ 模块测试 |
| `tests/gateway/` (156 文件) | gateway/ 模块测试 |
| `tests/hermes_cli/` (109 文件) | hermes_cli/ 模块测试 |
| `tests/run_agent/` (45 文件) | run_agent.py 测试 |
| `tests/cli/` (39 文件) | cli.py 测试 |
| `tests/acp/` (9 文件) | ACP 适配器测试 |
| `tests/cron/` (7 文件) | cron 模块测试 |
| `tests/environments/` (1 文件) | 环境测试 |
| `tests/skills/` (6 文件) | 技能测试 |
| `tests/integration/` (8 文件) | 集成测试 |
| `tests/e2e/` (3 文件) | 端到端测试 |
| `tests/plugins/` (3 文件) | 插件测试 |
| `tests/honcho_plugin/` (5 文件) | Honcho 插件测试 |
| `tests/fakes/` (2 文件) | 测试替身 |

**关键约定：**
- `conftest.py` 的 `_isolate_hermes_home` autouse fixture 重定向 HERMES_HOME 到临时目录
- 必须使用 `scripts/run_tests.sh` 运行测试以确保 CI 一致性

---

### 12. `nix/` — Nix 打包（6 个文件）

**文件类型：** Nix 表达式文件

| 文件 | 功能 |
|------|------|
| `packages.nix` | 包定义 |
| `python.nix` | Python 环境构建 |
| `devShell.nix` | 开发 Shell |
| `nixosModules.nix` | NixOS 系统模块 |
| `checks.nix` | 检查 |
| `configMergeScript.nix` | 配置合并脚本 |

---

### 13. `web/` — 前端 Web 界面

**文件类型：** TypeScript + Vite + React 项目

**职责：** Hermes 的 Web 管理界面，独立于 Python 后端

---

### 14. `website/` — 项目文档站

**文件类型：** Docusaurus 项目（TypeScript + MDX）

**职责：** 项目公开文档网站

---

### 15. `tinker-atropos/` — RL 训练子模块

**文件类型：** Python 项目（git 子模块）

**职责：** 基于 Atropos 的 RL 训练框架，包含训练配置和启动脚本

---

### 16. `acp_registry/` — ACP 注册信息

**文件类型：** JSON + SVG

| 文件 | 功能 |
|------|------|
| `agent.json` | ACP 代理元数据注册 |
| `icon.svg` | 代理图标 |

---

### 17. `scripts/` — 运维脚本

**文件类型：** Shell/Python 脚本

| 关键文件 | 功能 |
|---------|------|
| `run_tests.sh` | 测试运行封装（强制 CI 环境一致性） |
| `install.sh` / `install.ps1` / `install.cmd` | 安装脚本 |
| `release.py` | 发布流程脚本 |
| `build_skills_index.py` | 技能索引构建 |
| `sample_and_compress.py` | 采样压缩工具 |
| `hermes-gateway` | Gateway 服务脚本 |
| `whatsapp-bridge/` | WhatsApp 桥接脚本 |
| `discord-voice-doctor.py` | Discord 语音诊断 |

---

### 18. `docs/` — 项目文档

**文件类型：** Markdown/HTML 文档

**职责：** 包含迁移指南、计划、皮肤文档、规格说明等

---

### 19. `docker/` — Docker 配置

**文件类型：** Docker 相关文件

| 文件 | 功能 |
|------|------|
| `SOUL.md` | Docker 容器的代理人格定义 |
| `entrypoint.sh` | 容器入口脚本 |

---

### 20. `datagen-config-examples/` — 数据生成配置示例

**文件类型：** YAML/JSONL/Shell 配置示例

---

### 21. `packaging/` — 打包配置

**文件类型：** Homebrew formula 等

---

## 三、目录依赖关系图

```
                    ┌─────────────────────────────────────────┐
                    │         用户入口 (Entry Points)           │
                    │  hermes_cli/main.py → cli.py → run_agent.py  │
                    │  gateway/run.py     batch_runner.py         │
                    │  acp_adapter/       rl_cli.py               │
                    └──────────────┬──────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
              ▼                    ▼                     ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │   agent/         │  │  model_tools.py  │  │  gateway/        │
    │  (代理核心逻辑)   │  │  (工具编排层)     │  │  (消息平台网关)   │
    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
             │                    │                     │
             │           ┌────────┴────────┐            │
             │           ▼                 ▼            │
             │  ┌──────────────┐  ┌──────────────┐     │
             │  │ tools/       │  │ toolsets.py   │     │
             │  │ (工具实现)    │  │ (工具集定义)  │     │
             │  └──────┬───────┘  └──────────────┘     │
             │         │                               │
             └─────┬───┘                               │
                   │                                   │
                   ▼                                   ▼
    ┌──────────────────────────────────────────────────────────┐
    │              基础设施层 (Infrastructure)                   │
    │  hermes_constants.py ← hermes_logging.py ← hermes_time.py│
    │  hermes_state.py     ← utils.py                         │
    └──────────────────────────────────────────────────────────┘
                         ▲
                         │
              ┌──────────┴──────────┐
              │                     │
    ┌─────────┴────────┐  ┌────────┴─────────┐
    │  hermes_cli/      │  │  cron/           │
    │  (CLI 子命令/配置) │  │  (定时任务)      │
    └──────────────────┘  └──────────────────┘

    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  environments/   │  │  plugins/       │  │  skills/        │
    │  (RL 训练环境)   │  │  (插件系统)     │  │  (技能定义)     │
    └─────────────────┘  └─────────────────┘  └─────────────────┘

    ┌─────────────────┐  ┌─────────────────┐
    │  acp_adapter/    │  │  tinker-atropos/ │
    │  (编辑器集成)    │  │  (RL 训练框架)   │
    └─────────────────┘  └─────────────────┘
```

## 四、核心依赖链

```
hermes_constants.py  (零依赖 — 依赖链最底层)
        ↑
utils.py, hermes_logging.py, hermes_time.py, hermes_state.py
        ↑
tools/registry.py  (零项目内依赖 — 只用 stdlib)
        ↑
tools/*.py  (各自调用 registry.register()，引用 hermes_constants/agent/utils)
        ↑
toolsets.py  (定义工具集，被 model_tools 引用)
        ↑
model_tools.py  (触发工具发现，提供公共 API)
        ↑
run_agent.py  (AIAgent 核心循环)
        ↑
cli.py  (HermesCLI 交互式终端)

agent/  ← 引用 hermes_constants、utils、hermes_cli.auth/config
tools/  ← 引用 hermes_constants、agent.auxiliary_client/redact
gateway/ ← 引用 hermes_constants、hermes_cli.config、utils
hermes_cli/ ← 引用 hermes_constants、agent.credential_pool/models_dev、tools.*、gateway.*
cron/ ← 引用 hermes_constants、hermes_cli.config、hermes_time
environments/ ← 引用 model_tools、tools.*
acp_adapter/ ← 引用 hermes_constants
plugins/ ← 引用 hermes_constants
```

## 五、重要架构规则

1. **`hermes_constants.py` 是唯一零依赖模块** — 可被任何模块安全导入，不会产生循环依赖
2. **工具自注册模式** — `tools/*.py` 在导入时调用 `registry.register()`，`model_tools.py` 通过 `discover_builtin_tools()` 触发自动发现
3. **禁止在工具 Schema 中硬编码跨工具引用** — 工具可能因缺少 API Key 或禁用而不可用，动态引用应在 `get_tool_definitions()` 中添加
4. **路径必须使用 `get_hermes_home()`** — 硬编码 `~/.hermes` 会破坏多 Profile 支持
5. **提示缓存不可破坏** — 不得在对话中途更改上下文、工具集、记忆或重建系统提示
6. **测试必须使用 `scripts/run_tests.sh`** — 确保 CI 环境一致性（unset 凭证、TZ=UTC、4 workers）
7. **禁止使用 `simple_term_menu`** — 在 tmux/iTerm2 中有渲染 bug，使用 curses 代替
8. **Spinner 中禁止使用 `\033[K`** — 在 prompt_toolkit 的 patch_stdout 下会泄漏为字面文本
