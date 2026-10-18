<div align="center">
  <img src="assets/images/banner.jpeg" alt="Octop Harness" width="600" />
  <h1>Octop Harness</h1>
</div>

## Request and run observations

`HarnessAgentConfig.request_snapshot_sink` receives bounded final model requests
and their usage receipts. `run_event_sink` receives correlated tool and plan
lifecycle events with names, references and bounded plan items; file contents,
command input and tool output bodies stay outside that ledger. Both callbacks
are runtime-only and are omitted from serialized agent configuration. The host
owns authentication, persistence and inspection of these observations.

Snapshots redact complete values before truncating text or identifiers. At most
256 component details are retained; `component_count`, `component_counts` and
`estimated_tokens_by_kind` still describe the complete request. Truncation flags
distinguish omitted details and bounded metadata from complete observations.
`ChatRequest` strips internal summary, skill and recall markers from external
messages; runtime-generated and checkpointed middleware messages retain them.

With separate project execution, snapshots classify marked conversation summaries
as injected context and attach actual summarizer transformation receipts. Counts
before/after are message estimates, not provider billing; reusing an old summary
does not count as a fresh compaction. Successful manual compaction emits a run
receipt with the retained history reference and message estimates.

## Separate expert storage and project execution

Set `HarnessAgentConfig.execution_backend` to a backend spec or live backend instance
when project tools must operate independently of the expert's content store.
`backend` and `workspace_dir` continue to locate internal expert materials and
runtime persistence. `agent.agent_store` exposes that storage facade;
`agent.project_backend` exposes project file and execution operations.

The graph addresses internal materials through `/.agent/` and sends other file
operations and commands to the project backend. Skills, memory sources and
compaction history retain their storage location. A host can supply a backend
that resolves the execution target per request; it must preserve authenticated
ownership and must not mutate a shared agent's target between concurrent calls.

Backend specs round-trip through configuration serialization. Live backend
instances are runtime dependencies and are omitted from serialized config.

OpenAI models that require Responses can set `ModelConfig(use_responses_api=True)`.
Chat Completions remains the default, and this setting does not enable hosted tool
search. OpenCode Go presets carry the appropriate per-model value.

<p align="center">
  <strong>A production-grade agent runtime engineered from the Harness Engineering philosophy.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white" /></a>
  <a href="https://github.com/TencentCloud/octop-harness/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green" /></a>
  <a href="https://pypi.org/project/octop-harness/"><img src="https://img.shields.io/pypi/v/octop-harness" alt="PyPI" /></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Code Style: Ruff" src="https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff&logoColor=white" /></a>
  <a href="https://github.com/TencentCloud/octop-harness"><img alt="GitHub stars" src="https://img.shields.io/github/stars/TencentCloud/octop-harness?style=social" /></a>
</p>

<p align="center">
  <a href="#what-is-octop-harness">What is Octop Harness?</a> ·
  <a href="#why-octop-harness">Why Octop Harness?</a> ·
  <a href="#how-to-use">How to Use</a> ·
  <a href="#documentation">Documentation</a>
</p>

<p align="center">
  <b>English</b> · <a href="README_CN.md">中文</a>
</p>

---

**octop-harness** is a production-grade agent runtime built on the harness engineering theory. At its core it is a thin, battle-tested **encapsulation and engineering layer around [deepagents](https://github.com/langchain-ai/deepagents)** — we take the elegant `create_deep_agent` primitive from deepagents and package it with everything you need to run agents in production: multi-provider model routing, persistent memory, browser/search tools, a multi-agent registry, pluggable storage backends, and a terminal CLI.

> 💜 **A tribute to deepagents.** octop-harness stands on the shoulders of [deepagents](https://github.com/langchain-ai/deepagents) (by the LangChain team). Our `HarnessAgent` ultimately delegates to `deepagents.create_deep_agent`, and everything else — model routing, backends, tiered memory, the agent registry, CLI, and ACP — is the engineering we added on top. deepagents gives us the agentic foundation; Harness gives it a home for production.

## What is Octop Harness?

| | Feature | Description |
|---|---------|-------------|
| 🧩 | **Built on deepagents** | A faithful, production-focused facade over `deepagents.create_deep_agent` — kudos to the deepagents project |
| 🔀 | **Model routing** | `ChatModelFactory` with OpenAI, Anthropic, AWS Bedrock, and 17 editable provider presets |
| 🧠 | **Persistent memory** | [octop-memory](https://github.com/TencentCloud/octop-memory) middleware with tiered L0→L3 distillation |
| 🛠️ | **Rich toolset** | Browser automation, web search, file ops, sub-agents, and MCP tools out of the box |
| 🗂️ | **Multi-agent registry** | `AgentManager` runs many isolated agents in one process |
| 💾 | **Pluggable backends** | Local disk, S3, COS, or PostgreSQL as the agent workspace |
| 🔌 | **ACP integration** | A stdio ACP server so IDE / terminal AIs can drive your agent |
| 🔒 | **Safety built-in** | Tool guardrails, filesystem permissions, and PII redaction |
| 💬 | **Teams** | Peer inbox; `peer_invoke_mode` selects sync / async / both |
| ⌨️ | **Terminal CLI** | Interactive chat, provider/skill config, and agent management |

## Why Octop Harness?

octop-harness is a library (not an app) that turns deepagents into a deployable runtime. A single `HarnessAgentManager` owns a registry of agents; each agent wires together a model, a set of tools/skills, a memory backend, and a LangGraph checkpoint — all assembled through the `harness` layering discipline (L0 I/O helpers → L1 workspace facade → L2 business logic → L3 assembly).

> Harness Agent's design goal: let you build a production-grade agent from deepagents in a few lines of code, while keeping model choice, memory, storage, and safety swappable without rewriting your agent.

### Core Technology

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.12+ |
| **Agent core** | [deepagents](https://github.com/langchain-ai/deepagents) (`create_deep_agent`) |
| **Graph runtime** | LangGraph + SQLite checkpoint |
| **Model routing** | `ChatModelFactory` (OpenAI / Anthropic / Bedrock + 17 presets) |
| **Memory** | [octop-memory](https://github.com/TencentCloud/octop-memory) middleware |
| **Browser / search** | [octop-browser](https://github.com/TencentCloud/octop-browser) + web search |
| **Workspaces** | `BackendWorkspace`: local / S3 / COS / PostgreSQL |
| **ACP** | agent-client-protocol |
| **Build / quality** | hatchling · ruff · mypy · pytest |

### Features

#### Agent runtime
- `HarnessAgentManager` — a registry that creates, lists, and streams many agents in one process.
- `HarnessAgent` — exposes `call`, `stream`, `stream_events`, `aget_history`, `aappend_messages`, and `cancel`.
- Checkpointing via LangGraph `AsyncSqliteSaver` (or a octop-memory saver) for resumable chats.

#### Model routing & providers
- `ChatModelFactory` resolves a model name to the right client.
- First-class providers: **OpenAI** (incl. OpenAI-compatible via `base_url`), **Anthropic**, and **AWS Bedrock** (optional `[bedrock]` extra).
- 17 user-editable provider presets, including Hunyuan, Kimi, GLM, DeepSeek, MiniMax, and Moonshot.

#### Memory (tiered)
- Memory-enabled agents expose `memory_search`, `memory_get`, `memory_list`,
  `memory_write`, `memory_update`, and `memory_delete`. Writes use the agent's
  bound namespace. Update/delete require an `atom/<id>.md` path, the exact
  current assertion (`expected_content`), and a verified reason. Update creates
  a successor; delete retires the fact without erasing history. The returned
  path can be read back to verify a correction.
- Corrected facts, their retired raw evidence and affected summaries awaiting
  regeneration are excluded from new agent recall. Raw evidence remains in
  host audit/history; existing conversation/checkpoint messages are preserved.
- Each turn is captured as **L0 raw events**, then distilled asynchronously into **L2 atoms** (`AtomCard`s) and **L3 entity pages**.
- Powered by the octop-memory middleware; memory travels with the workspace.
- Recall is frozen once per user turn and appended to the model-facing user message,
  keeping the system prompt stable. Checkpoints preserve recall snapshots for exact
  replay while chat history and capture retain the original user text.

#### Tools & skills
- Built-in tools: browser (octop-browser), web search, files, ``ask_user_question`` for
  respond-only human decisions, optional provider-neutral image/video generation, sub-agents,
  and MCP tools. Set ``ask_user_enabled=False`` for unattended agents.
- Skills come from the deepagents skills middleware; manage them per agent via the CLI.
- Optional ``system_files_path`` (e.g. ``.octop``) keeps skills, sessions, ``.env``,
  and sqlite under a workspace subdir while persona markdown stays at the root.

Enable provider-neutral `generate_image` and `generate_video` tools with Volcengine Ark:

```python
from octop_harness import HarnessAgentConfig, MediaGenerationConfig

config = HarnessAgentConfig(
    # ...providers/default_model...
    media_generation=MediaGenerationConfig(api_key_env="ARK_API_KEY"),
)
```

Generated files are stored under `generated/images/` and `generated/videos/` in the agent workspace. The tools
are deferred by default and become visible through the configured tool-search strategy only when needed.
Expected provider failures return a structured, model-visible error envelope with retry guidance instead of
terminating the agent stream.

For separate image and video routes, configure named providers. Built-in adapters cover Volcengine Ark
(Seedream/Seedance), Alibaba Cloud Model Studio (Wan), and MiniMax (image-01/H3):

```python
from octop_harness import HarnessAgentConfig, MediaGenerationConfig, MediaProviderConfig

config = HarnessAgentConfig(
    # ...providers/default_model...
    media_generation=MediaGenerationConfig(
        providers=[
            MediaProviderConfig(
                id="ark-images",
                provider="volcengine",
                api_key_env="ARK_API_KEY",
                video_enabled=False,
            ),
            MediaProviderConfig(
                id="minimax-videos",
                provider="minimax",
                api_key_env="MINIMAX_API_KEY",
                image_enabled=False,
            ),
        ],
        default_image_provider="ark-images",
        default_video_provider="minimax-videos",
    ),
)
```

Advanced hosts can inject a `MediaManager` into `build_media_generation_tools` and register a constrained
custom adapter without adding arbitrary protocol or authentication fields to declarative configuration.

#### Multi-agent & collaboration
- Multiple isolated agents per process via `AgentManager` + `registry`.
- **Teams** peer inbox for agent-to-agent messaging. ``peer_invoke_mode``
  selects the ``ask_agent`` surface (``sync`` / ``async`` / ``both``);
  request-scoped overrides may tighten it but cannot add background dispatch.
  All surfaces validate the same dispatch fields and reject unknown arguments.
  `DispatchOptions` defaults to `target_policy="member_default"`; choose
  `inherit` deliberately, or `explicit` with `device_id` / `target_kind="local"`.
  A host must authorize target and resource requests. Up to 32 unique registered
  `resource_ids` request bounded access; paths and owner claims grant nothing.
  `DispatchReceipt` separates transport status from `task_outcome` (unknown
  unless a structured result supplies it) and identifies the actual resolved
  target, invocation and child thread. Host `bind_peer_session(prepare=...,
  lifecycle=...)` hooks persist acceptance before execution and resolve the
  session; an accepted callback returning an existing receipt skips replay.
  `call_peer` and awaited `asubmit_peer` accept a trusted `invocation_key` for
  idempotency and `retry_of` for a host-authorized new attempt. Real tools use
  the injected runtime tool-call ID; each invocation has an isolated fallback
  child thread/session, and sync calls keep `job_id=None`.
  The obsolete synchronous `submit_peer` fails with `use await asubmit_peer`.
  `await team.acancel_peer(invocation_id)` persists a cancellation request
  before interrupting queued or active sync/background work. A task cancelled
  after entering the child reports unknown until its executor confirms the
  external work stopped; a late request preserves an already completed result.
  The runtime retains at most 256 completed full results when a lifecycle host
  is bound; older replay uses the host's durable receipt. Without that host,
  compact fingerprint/receipt tombstones remain for the manager's lifetime
  after large result bodies are released, so old keys still cannot execute again.
- **ACP** stdio server so external IDE / terminal AIs can invoke your agent.

#### Safety
- `SecurityPolicy` with tool guardrails, filesystem permissions (`FilesystemPermission`), and PII redaction middleware.
- PII detection covers provider API keys, mainland China mobile numbers (common 13-19 prefixes, optional `+86` / `0086`), and 18-character resident IDs with a known province code, 1900-2099 date, and checksum. Existing `pii_strategy` and `pii_surfaces` apply; legacy 15-digit IDs and other countries' numbers are out of scope.

#### CLI
`octop-harness` provides: `init`, `chat`, `agent`, `config` (e.g. `config provider add`), `skill`, and `update`.

## How to Use

### Prerequisites
- **Python 3.12+**
- A model provider API key (OpenAI / Anthropic / Bedrock / compatible)

### 1. Install

```bash
# Core SDK — agent runtime, model routing, tools, skills, backends
pip install octop-harness

# With the terminal CLI — interactive chat, config, skill management
pip install octop-harness[cli]

# Multiple extras — comma-separated inside one pair of brackets (quote for the shell):
pip install 'octop-harness[object-storage,desktop]'
pip install 'octop-harness[cli,all]'
```

Optional dependency extras (install only what you need; missing extras fail at use-time with an install hint):

| Extra | What it adds |
|-------|----------------|
| `cli` | Terminal CLI (`octop-harness`) |
| `bedrock` | AWS Bedrock provider |
| `object-storage` | Tencent COS + Alibaba OSS + Huawei OBS SDKs |
| `desktop` | Desktop screenshot / input (`mss`, `pynput`, `pillow`) |
| `web-search-all` | All web-search backends (Tavily / Brave / Google) |
| `remote-backends` | Postgres workspace backend (`psycopg`). S3 uses the bundled boto3 backend. |
| `observability` | Langfuse |
| `acp` | ACP agent runner |
| `all` | All library feature extras above (**excludes** `cli`; use `[cli,all]` for both) |

### 2. Initialize & configure

```bash
octop-harness init
octop-harness config provider add   # choose a provider and paste your key
```

### 3. Chat

```bash
octop-harness chat
```

### Programmatic use

```python
from octop_harness import HarnessAgentManager, HarnessAgentConfig, ProviderConfig, ChatRequest

manager = HarnessAgentManager()
agent: HarnessAgentConfig = manager.create_agent(
    name="assistant",
    provider=ProviderConfig(name="openai", api_key="sk-..."),
    model="gpt-4o",
)
request = ChatRequest(message="Summarize the Harness theory in one paragraph.")
async for chunk in agent.stream(request):
    print(chunk.delta, end="")
```

### Progressive tool loading

Large, low-frequency tool sets can be hidden until the model needs them. The
default `client` mode registers an ordinary `tool_search` function, so it works
with any Chat Completions-compatible model that supports function calling. A
deferred tool remains visible by its real name and short description, while its
parameter schema is replaced by a lightweight reference. A search result is
appended to the conversation, the matched full schemas replace their references
on the next model step, and the loaded set persists for the thread. The selected
tool is still called by its real name through the normal DeepAgents ToolNode,
security guard, and interrupt policy.

```python
from octop_harness import HarnessAgentConfig, ProviderConfig

provider = ProviderConfig(
    id="openai",
    base_url="https://openai-compatible.example/v1",
    api_key="sk-...",
    protocol="openai",
)

config = HarnessAgentConfig(
    providers=[provider],
    default_model="openai/your-function-calling-model",
    tools=[generate_image, generate_video, generate_3d_asset],
    deferred_tools=frozenset(
        {"generate_image", "generate_video", "generate_3d_asset"},
    ),
    defer_mcp_tools=True,
    tool_search_mode="client",  # default; no Responses API required
)
```

Set `tool_search_mode="native"` to use OpenAI Responses hosted `tool_search`
or Anthropic hosted `defer_loading` / `tool_reference`. Native mode additionally
requires `ModelConfig.native_tool_search=True`; unsupported routed models use
the configured `tool_search_fallback`. Set the mode to `eager` to disable
progressive loading.

## Documentation

- [What is Octop Harness?](#what-is-octop-harness)
- [Why Octop Harness?](#why-octop-harness)
- [How to Use](#how-to-use)
- **Reference**
  - [CLI reference](#cli-reference)
  - [Project layout](#project-layout)
  - [Development](#development)
- **Project Info**
  - [Contributing](#contributing)
  - [Related projects](#related-projects)
  - [License](#license)

### CLI reference

| Command | Description |
|---------|-------------|
| `octop-harness init` | Bootstrap a workspace and config |
| `octop-harness chat` | Interactive chat with an agent |
| `octop-harness agent` | Create, list, and manage agents |
| `octop-harness config` | Provider / model configuration (`config provider add`) |
| `octop-harness skill` | Enable / disable per-agent skills |
| `octop-harness update` | Check for and install updates |

### Project layout

```
src/octop_harness/
  agent.py          facade over deepagents.create_deep_agent
  manager.py        AgentManager — multi-agent registry
  config/           configs, provider & model presets
  llm/factory.py    ChatModelFactory — model routing
  backends/         BackendWorkspace — local / S3 / COS / Postgres
  builtin/          tools + skills + seed files
  middleware/       model_router, skill_filter, tool_guard, memory, pii, ...
  memory/           MemoryRuntime (octop-memory wrapper)
  protocols/        langgraph / openai / mcp streaming
  acp/             ACP stdio server
  teams/           peer inbox
  cli/             terminal CLI
```

### Development

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
make install          # pip install -e ".[cli,dev]"
make all              # lint + typecheck + test
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run `make all` before submitting
4. Open a Pull Request against `main`

Module boundaries and coding conventions: [AGENTS.md](AGENTS.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) for branching, PR, and release details (`release/*` → `main` auto-publishes to PyPI).

## 🔗 Related projects

| Project | Description |
|---------|-------------|
| [deepagents](https://github.com/langchain-ai/deepagents) | The agentic foundation Harness Agent wraps — ❤️ tribute |
| [octop-memory](https://github.com/TencentCloud/octop-memory) | Memory system behind the tiered recall |
| [octop-browser](https://github.com/TencentCloud/octop-browser) | CDP browser automation used by the agent |
| [octop-gateway](https://github.com/TencentCloud/octop-gateway) | Multi-platform IM channel bridge |
| [Octop](https://github.com/TencentCloud/Octop) | The self-hosted assistant that composes the Harness stack |

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## ✨ Contributors

Thanks to all contributors:

<a href="https://github.com/TencentCloud/octop-harness/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TencentCloud/octop-harness" />
</a>
