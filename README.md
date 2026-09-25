# sigma-verify

[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

An MCP server that lets Claude Code agents check their findings against a model other than Claude. It wraps OpenAI, Google Gemini, and models served by [Ollama](https://ollama.com) (local and cloud) behind a small set of tools, and tags every result with the provider and model that produced it.

It is built on [hateoas-agent](https://github.com/coloradored13/hateoas-agent) and is used by the [sigma](https://github.com/coloradored13/sigma) Claude Code plugin (sigma-review, sigma-build), but it works with any MCP client.

## Tools

The server starts with one gateway tool. Calling it unlocks the rest.

| Tool | What it does |
|---|---|
| `init` | Reports which providers are available, which aren't, and why. Call it first. |
| `verify_finding` | One model assesses a finding: agree / disagree / partial / uncertain, with reasoning, confidence, and counter-evidence. |
| `cross_verify` | Runs `verify_finding` across several providers at once and reports the level of agreement. |
| `challenge` | One model plays devil's advocate against a claim: counter-argument, logical gaps, evidence needed, vulnerability. |
| `get_models` | Lists available providers and models. |
| `check_quotas` | Summarizes pricing and quota notes for every provider, including unavailable ones. |

### Provenance tags

Every result names its source, so it's clear which model said what:

- Each result carries `provenance_tag: "|source:external-<provider>-<model>|"`.
- Agents record a successful check as `XVERIFY[<provider>:<model>]`, for example `XVERIFY[gpt-oss:gpt-oss:120b-cloud]: agree(high)`.
- A failed call returns an `XVERIFY-FAIL[<provider>:<model>]` string in its `sigma` field with an error class (`auth-error`, `rate-limit`, `timeout`, `network-error`, and others). That marks a verification gap. It does not mean the model disagreed.
- `cross_verify` reports `coverage: "partial"` whenever any provider fails, so a partial run isn't mistaken for full cross-model agreement.

## Providers

| Provider | Default model | Type | Requires |
|---|---|---|---|
| `openai` | `gpt-5.4` (reasoning: `gpt-5.4-pro`) | API | `OPENAI_API_KEY` |
| `google` | `gemini-3.1-pro-preview` | API | `GOOGLE_AI_API_KEY` |
| `llama` | `llama3.1:8b` | local | Ollama + `ollama pull llama3.1:8b` |
| `gemma` | `gemma4:e4b` | local | Ollama + `ollama pull gemma4:e4b` |
| `nemotron-nano` | `nemotron-3-nano:4b` | local | Ollama + `ollama pull nemotron-3-nano:4b` |
| `qwen-local` | `qwen3.5:4b` | local | Ollama + `ollama pull qwen3.5:4b` |
| `gpt-oss` | `gpt-oss:120b-cloud` | free cloud | Ollama + `ollama signin` |
| `nemotron` | `nemotron-3-super:cloud` | free cloud | Ollama + `ollama signin` |
| `deepseek` | `deepseek-v4.1-flash:cloud` | paid cloud | Ollama + `ollama signin` + credits |
| `mistral` | `mistral-large-3:675b-cloud` | paid cloud | Ollama + `ollama signin` + credits |
| `glm` | `glm-5.3:cloud` | paid cloud | Ollama + `ollama signin` + credits |
| `kimi` | `kimi-k3:cloud` | paid cloud | Ollama + `ollama signin` + credits |
| `anthropic` | `claude-opus-5-5` | API | `ANTHROPIC_API_KEY`, and only when named explicitly (see below) |

Ollama's free and paid tiers can change. Check ollama.com for current terms.

Local models run on your machine at no API cost. Cloud models run on ollama.com through your local Ollama install. `init` checks that Ollama is reachable and that local models have been pulled. It can't cheaply check whether you're signed in, so if you aren't, a cloud provider will show as available and then fail its first call with an `auth-error`.

Any model can be overridden with `<PREFIX>_MODEL`. The endpoint can be overridden with `<PREFIX>_BASE_URL`. `llama`, `gemma`, `gpt-oss`, and `nemotron` can also be routed through OpenRouter with `<PREFIX>_PROVIDER=openrouter` and `OPENROUTER_API_KEY`. [`.env.example`](.env.example) lists every variable.

### Anthropic is excluded by default

The callers are Claude agents, and Claude checking Claude is not cross-model verification. So `anthropic` is left out of every default selection: the `cross_verify` default set, the provider `verify_finding` and `challenge` pick when none is named, and the providers `init` lists as available. It is used only when:

- the caller names it (`provider="anthropic"`, or `providers="anthropic,..."`), or
- `SIGMA_VERIFY_ALLOW_ANTHROPIC=1` is set.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
claude mcp add sigma-verify --scope user -- \
  uvx --from git+https://github.com/coloradored13/sigma-verify sigma-verify
```

To add API-key providers, pass keys with `-e`:

```bash
claude mcp add sigma-verify --scope user \
  -e OPENAI_API_KEY=sk-... \
  -e GOOGLE_AI_API_KEY=... \
  -- uvx --from git+https://github.com/coloradored13/sigma-verify sigma-verify
```

Then, in Claude Code, ask an agent to call `init`, or run `/mcp` to confirm that the server is connected.

### Ollama setup (optional)

Ollama-backed providers need no API key.

```bash
ollama serve                 # or start the Ollama app
ollama pull llama3.1:8b      # any local model from the table
ollama signin                # only for cloud models
```

sigma-verify doesn't start Ollama for you by default. If Ollama isn't running, its providers are reported as unavailable. To have `init` (and server startup) run `ollama serve` when Ollama is installed but not running, set `SIGMA_VERIFY_AUTOSTART_OLLAMA=1`, for example with `-e SIGMA_VERIFY_AUTOSTART_OLLAMA=1` on `claude mcp add`.

## With no providers

You don't need any provider configured. With none available, `init` reports `no_providers` and lists what each provider needs. sigma-review and sigma-build still run, and findings just stay untagged. An untagged finding is neutral: it hasn't been checked against another model, and nothing counts against it.

## Development

```bash
uv venv .venv
uv pip install -p .venv -e '.[dev]'
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src tests
```

The test suite runs offline. A shared fixture fakes the Ollama probe and blocks outbound TCP connections.

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
