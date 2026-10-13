# Model catalog refresh (2026-10-13)

The bundled `src/octop_harness/providers/provider_template.json` remains the
single source for provider presets. Existing application configurations are not
overwritten by a library upgrade.

- [DeepSeek pricing and model table](https://api-docs.deepseek.com/quick_start/pricing/):
  V4.1 Flash uses `deepseek-flash` on the native API, accepts text and images,
  and documents a 1M input / 384K output limit. V4 Flash remains a compatibility
  alias; V4 Pro keeps its existing route.
- [OpenCode Go discovery endpoint](https://opencode.ai/inference/go/v1/models):
  the endpoint lists both `deepseek-flash` and `deepseek-v4.1-flash`, plus the
  stable model IDs added in this release. Removed `grok-4.5`, which is absent
  from discovery. Experimental/preview/free-only entries are not new defaults.

The OpenCode discovery response does not include token limits or input
modalities. Newly listed models retain unknown limits unless documented
separately; DeepSeek Flash metadata follows the vendor documentation. The
existing Compatible / Anthropic protocol split is preserved. The recommended
base URL now uses `/inference/go`; saved application URLs are not overwritten.
[OpenCode's endpoint table](https://opencode.ai/v2/docs/console/go/) requires
Responses for Grok, Luna and Muse; those presets carry `use_responses_api=true`.
Chat Completions remains the default for the other compatible models.

The CkEFFAF fork publishes `1.0.2.post1` wheels to GitHub Releases. It does not
publish to TencentCloud's PyPI project. Octopus pins the release wheel URL and
lockfile hash, so deployments receive the same catalog without a second JSON.
