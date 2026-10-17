# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 新增

- 记忆工具新增 `memory_list`、`memory_write`、`memory_update`、`memory_delete`，限定当前专家的事实卡；更正生成后继、删除执行停用，并保留审计和原始历史。更正要求匹配当前原文及提供原因，工具检索和新回合自动召回过滤失效事实、相关旧来源及待重建摘要。

### 修复

- 每次专家调用标记独立运行身份，流式回答和审批恢复过滤嵌套专家的消息，避免子专家文本、推理、工具片段及用量混入父专家输出；子专家结果仍保留在工具回执中。
- 专家资料按当前工作区授权，恢复管理端敏感主目录下的记忆文件读写，并将其与设备用户的系统权限分开。隐藏路径、递归搜索结果、目录删除、符号链接与 Windows 路径别名均受检查；认证、会话及已配置的数据库存储保持由管理端维护。缺失的可选旧技能目录不再产生加载噪音。

## [1.0.2.post2] - 2026-10-15

### 新增

- 支持通过 `execution_backend` 分离专家内部存储与项目执行后端，公开 `agent.agent_store` 和 `agent.project_backend`；内部资料通过 `/.agent/` 路由，技能、记忆来源和压缩历史保留在专家存储中。
- 提供可选的 `request_snapshot_sink`，记录中间件处理后实际发往模型的有界请求快照、用量回执，以及参考文件、技能目录、固定技能、记忆召回、文件读取结果和会话摘要的来源归属；快照文本执行 PII 脱敏，媒体正文与模型凭据不进入快照。
- 提供可选的 `run_event_sink`，关联工具、计划和任务分派的运行回执及检索来源；后台分派受理与子任务实际结果分别记录，工具语义错误不会被误报为成功。观察回调仅在运行时使用，不写入序列化配置。
- 记录摘要中间件实际生成的新摘要及手动压缩回执，保留历史引用和压缩前后的消息数量、估算 token 数；复用既有摘要不会被计为一次新压缩，估算值不作为供应商计费用量。

### 修复

- 独立执行后端的文件与命令工具保留目标端原生路径，支持 Windows 盘符、POSIX 绝对路径和相对父目录路径；文件操作错误以失败工具消息返回，避免错误的虚拟路径校验或改写。
- 修复工作区原生绝对路径解析和目录移入自身子目录的校验；本地执行检查环境变量路径时将权限错误视为路径不可用，避免启动失败。
- Agent 关闭等待进行中的调用和流式输出结束，再释放模型、记忆和检查点资源；同步关闭 SQLite 时由连接工作线程完成清理，避免线程归属错误和事件循环退出时的关闭竞争。
- ACP 支持通过公开接口注册应用持有的服务，修复无默认事件循环及活动 prompt 取消时的退出清理；一个会话关闭失败不阻止其他会话释放。
- 请求快照先完整脱敏再截断，约束标识、元数据和明细数量，同时保留全量分类估算；外部消息不能伪造摘要、技能或记忆召回来源，压缩回执正确区分内容相同但标识不同的消息。

## [1.0.2.post1] - 2026-10-13

### 修复

- DeepSeek 预置增加 V4.1 Flash 的官方 `deepseek-flash` 路由及图片输入，保留 V4 Flash 兼容别名。
- OpenCode Go 预置按官方模型接口刷新，新增 V4.1 Flash、Grok 4.6/4.7、GLM 5.3、Luna、MiMo V2.6、Claude Haiku 5.5 和 Qwen 3.8；未确认的上下文/输出上限保持未知。
- Fork 发版只发布 GitHub 安装包；PyPI 发布保留给上游仓库。
- OpenCode Go 的 Grok/Luna/Muse 路由显式使用 Responses API，其他兼容模型继续使用 Chat Completions；测试连接和聊天共享该模型配置。

## [1.0.2] - 2026-10-09

### 修复

- 将 `deepagents` 下限提高到 `>=0.7.16`，接入上游同路径并行 `edit_file` / `write_file` 防护（langchain-ai/deepagents#6446）。
- 记忆辅助调用可单独配置读取超时。提炼输出预算写入 memory runtime 的 `extraction.max_tokens`（提取器总会自带预算，客户端兜底覆盖不到）。OpenAI 兼容端点在 `auto` 下对 Qwen3 / QwQ / DeepSeek-R / reasoner 默认关闭深度思考（TencentCloud/Octop#1360）。
- 小时增量 vacuum 预算提高到 5000 页，大批删除后的 freelist 可在数小时内回收，而不是数周。

## [1.0.1] - 2026-10-03

### 新增

- PII 可识别大陆手机号与 18 位居民身份证。
- 子树专家支持 `explicit_virtual_paths`，由调用方自行转换虚拟路径。
- 新增讯飞星辰 Token Plan 供应商预设。

### 修复

- 子 Agent 仅在真正重名时告警。
- 模型和路径错误回传给 Agent，不再整轮中止。
- 修复 S3 / Postgres workspace 与 deepagents 0.7 的兼容，文件不再摊平到存储根。
- Docker 每次执行刷新全局环境。
- 流式 thinking / text 块不再导致协议投影崩溃。

### 变更

- 发版时同步检查多语言 README 版本。
### Added

- PII protection detects mainland China mobile numbers (common 13-19
  prefixes, optional ``+86`` / ``0086``) and 18-character resident IDs with
  a known province code, 1900-2099 birth date, and a valid checksum.
  Existing ``pii_strategy`` / ``pii_surfaces`` apply; 15-digit IDs and other
  countries’ numbers are out of scope.
- Optional ``explicit_virtual_paths`` for subtree experts: shell command,
  environment, and stdout are left unparsed; callers convert one virtual
  absolute path through ``virtual_to_native_path``. Default stays off.
- Bundled provider templates: iFlytek Astron MaaS Token Plan preset
  `iflytek-astron-token-plan` (Spark-X2.5 first; Kimi-K2.6 and Qwen3.5/3.6 marked
  image-capable), with an `iflytek.svg` logo.

### Fixed

- Workspace subagent loading no longer logs a spurious ``Duplicate subagent
  name`` warning when the same ``.octop/agents/*.md`` definition is discovered
  through both the legacy ``agents/`` root and the canonical
  ``system_files_path`` root; the warning now only fires for genuine name
  collisions between different files.
- Leftover Windows absolute paths in filesystem tool calls (``D:\\octop-data\\…``)
  are rewritten onto the current storage root when the same suffix exists, and
  still soft-fail to the model when they do not — they no longer abort the turn
  as a generic model-call failure.
- S3 backend resolution always uses the bundled boto3 implementation, which
  speaks deepagents 0.7 (`WriteResult` / `ReadResult` / `ls` / `glob` /
  `grep`). The `deepagents-backends` 0.2 S3 client is no longer preferred —
  it still passes the removed `files_update` keyword and cannot list or read
  under 0.7. Community path-style aliases (`s3_force_path_style` /
  `force_path_style` / `path_style`) map to `addressing_style="path"`.
  Custom endpoints default to SigV4 + path-style, disable optional CRC
  checksums, and fall back ListObjectsV2 → ListObjects / GetObject when the
  store rejects a header or signature flavour.
- Postgres backend specs accept a libpq `connection_string` / `dsn` and drop
  unknown keys instead of raising `TypeError` on `PostgresConfig`.
- Postgres workspace I/O uses a bundled `psycopg` backend that speaks
  deepagents 0.7 (`WriteResult` / `ReadResult` / `ls` / `glob` / `grep`).
  Probe performs a real write→read→delete round-trip; the table is created
  on first use.
- COS / S3 / OSS / OBS / Postgres workspaces no longer flatten files onto
  the storage root. Agent files live under `/.octop/workspaces/<id>/` so
  listing `/` shows workspace folders, then the workspace tree. Host
  paths such as `~/.octop/agents/<id>` are remapped onto that virtual
  folder and are not stored as object keys.
- Docker execution reloads the global environment file even when a same-size edit or atomic replacement preserves its modification timestamp, avoiding stale injected values.
- Streamed content blocks (Anthropic thinking / text, OpenAI Responses) no
  longer crash chunk projection: list-typed ``AIMessageChunk.content`` is
  normalized per block (thinking → reasoning, text → token) instead of raising
  ``TypeError`` in ``ThinkSplitter.feed``. Plain ``str`` content is unchanged.

## [1.0.0] - 2026-09-24

### 新增

- 首个 1.0.0 正式版本发布到 PyPI。

### 变更

- 对齐 Octop：引入 `develop` 集成分支策略；禁止直推 `main`/`develop`；发版后由 `sync-main-to-develop.yml` 同步；新增 `/publish` skill（发版同步 CHANGELOG / README）。



### Fixed

- Built-in tools soft-fail instead of aborting the agent turn: ``web_fetch``
  (non-http(s) / oversized body), ``current_time`` (unknown timezone),
  ``desktop_screenshot`` (capture ``RuntimeError``), ``acp_runner`` (missing
  ``thread_id`` on close/stream), and local ``execute`` (non-positive timeout).
- Filesystem tools: ``Path ... outside root directory`` ``ValueError`` from
  deepagents ``FilesystemBackend`` is softened by ``FilesystemGuardMiddleware``
  (always mounted) into a ``ToolMessage(status="error")`` so ToolNode does not
  abort the agent turn; deny-path rules still apply when permissions are set.
