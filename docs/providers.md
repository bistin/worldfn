# LLM providers

Agents only ever see `Llm`. The provider behind it is chosen when the world is
built, so switching between a fake, a subscription, and an API key never
touches agent code.

| Cargo feature | Type | Auth | Endpoint |
|---|---|---|---|
| none | `FakeLlm` | none | none; deterministic, for tests |
| `codex` | `providers::CodexLlm` | ChatGPT subscription, via the official Codex CLI's saved login | `chatgpt.com/backend-api/codex/responses` (unofficial) |
| `codex-login` | `CodexLlm::from_login` | ChatGPT subscription, via worldfn's own login (`worldfn login codex`) | same |
| `openai-compat` | `providers::OpenAiCompatLlm` | API key | any OpenAI-compatible `/chat/completions`: DeepSeek, OpenAI, local servers |

Try either one with the `live` example:

```sh
cargo run --example live --features codex -- "your question"
cargo run --example live --features openai-compat -- "your question"
```

The example chooses the provider from `WORLDFN_PROVIDER` (`codex`,
`deepseek`, `openai`) and the model from `WORLDFN_MODEL`. No model name is
built in, because providers rename and retire models often.

The core crate still builds on Rust 1.85. The provider features pull in
`reqwest`, whose current dependency tree needs a newer toolchain; they are
tested on stable 1.94.

## `CodexLlm`: ChatGPT subscription (personal experiments only)

> **Read this first.** This uses an endpoint that OpenAI built for its own
> Codex clients. It is not a public API for third-party software. It can change
> or break without notice, and using it outside OpenAI's own tools may conflict
> with OpenAI's terms and put your account at risk. Keep it to personal,
> interactive-scale experiments. Do not use it in a service, for batch jobs, or
> for other people. For those, use an API key (`OpenAiCompatLlm::openai`).

There are two ways to sign in. Both end with the same requests.

### Option 1: worldfn's own login (feature `codex-login`)

This works like pi's `/login`. The Codex CLI is not needed.

```sh
cargo run --features codex-login --bin worldfn -- login codex            # browser
cargo run --features codex-login --bin worldfn -- login codex --device   # no browser (SSH, containers)
cargo run --features codex-login --bin worldfn -- status
cargo run --features codex-login --bin worldfn -- logout codex
# or install it: cargo install --path . --features codex-login --bin worldfn
```

- **Browser.** worldfn prints a sign-in URL on `auth.openai.com` and tries to
  open it. After you sign in, the browser redirects to
  `http://localhost:1455/auth/callback`, which worldfn serves while it waits.
  The flow uses PKCE, and worldfn ignores callbacks whose `state` does not
  match. If port 1455 is busy, or the browser runs on another machine, copy
  the URL of the page the browser ends on and paste it into the terminal.
- **Device code.** worldfn shows a code. Enter it at
  `auth.openai.com/codex/device` on any device. worldfn polls for up to 15
  minutes. If this reports "not enabled", device login is not available for
  your account; use the browser login.
- **Tokens** are saved to `~/.worldfn/auth.json`, or `$WORLDFN_HOME/auth.json`.
  The file has mode `0600` and its directory `0700`. worldfn never sees your
  password. `status` never prints tokens. Do not paste this file into chats or
  issues.
- **Refresh.** `CodexLlm::from_login(model)` refreshes the access token when
  it has less than 5 minutes left, writes the new one back, and refreshes only
  once when calls run concurrently. If the refresh token has been revoked, the
  error tells you to run `worldfn login codex` again.

```rust
world.provide_llm(CodexLlm::from_login("gpt-5.5")?.reasoning_effort("low"))?;
```

The OAuth parameters (client id, scopes, redirect, and device endpoints) are
the ones OpenAI's own Codex clients use, as pi's `openai-codex` login does.
This is the part most likely to break, and the part most likely to conflict
with OpenAI's terms. The note above applies with extra force.

### Option 2: reuse the official Codex CLI login (feature `codex`)

1. Log in with the **official** Codex CLI: `codex login`, and choose ChatGPT.
   It saves tokens to `$CODEX_HOME/auth.json`, or `~/.codex/auth.json` by
   default.
2. `CodexLlm::from_codex_home(model)` reads that file. It re-reads the file on
   every call, so when the Codex CLI refreshes the token, the next call picks it
   up.
3. In this mode worldfn **never refreshes tokens itself**, because the Codex
   CLI owns that file. When the saved token expires, the next call fails with a
   message telling you to run `codex` (which refreshes the login) or
   `codex login` again.

The examples use option 1 when `~/.worldfn/auth.json` has a login and the
`codex-login` feature is on, and option 2 otherwise.

### Requests

Requests follow the open-source Codex CLI and pi's `openai-codex` provider:
a Responses-API body with `store: false` and `stream: true`, the
`chatgpt-account-id` and `OpenAI-Beta: responses=experimental` headers, and a
server-sent-event reply. The client identifies itself honestly as
`originator: worldfn`.

Option 2 will not work if:

- you logged into Codex with an API key rather than ChatGPT; there are no
  ChatGPT tokens in `auth.json` in that case,
- Codex stores credentials in the OS keyring
  (`cli_auth_credentials_store = "keyring"`). Use the default file store,
- the network blocks `chatgpt.com`.

```rust
use worldfn::providers::CodexLlm;

world.provide_llm(
    CodexLlm::from_codex_home("gpt-5.5")?   // any model your plan offers in Codex
        .instructions("Answer briefly.")
        .reasoning_effort("low"),
)?;
```

## `OpenAiCompatLlm`: API keys (DeepSeek, OpenAI, local)

The standard, supported path. You pay per token on the provider's API plan.

```rust
use worldfn::providers::OpenAiCompatLlm;

// DeepSeek: reads DEEPSEEK_API_KEY, base URL https://api.deepseek.com
world.provide_llm(OpenAiCompatLlm::deepseek("deepseek-v4-flash")?)?;

// OpenAI API: reads OPENAI_API_KEY (separate billing from ChatGPT plans)
world.provide_llm(OpenAiCompatLlm::openai("gpt-5.5")?)?;

// Anything else that speaks /chat/completions, e.g. a local server
world.provide_llm(OpenAiCompatLlm::new("http://localhost:11434/v1", "unused", "llama3"))?;
```

Each call sends an optional `.system(..)` message plus one user message, and
returns `choices[0].message.content`.

## What gets sent

Both providers take the full `ChatRequest`:

| `ChatRequest` | OpenAI-compatible (`/chat/completions`) | Codex (Responses API) |
|---|---|---|
| `system` (or the provider's default) | `system` message | `instructions` |
| user / assistant text | `user` / `assistant` messages | `message` items (`input_text` / `output_text`) |
| assistant tool calls | `assistant.tool_calls` | `function_call` items |
| tool results | `tool` messages (`is_error` → `Error: ` prefix) | `function_call_output` items |
| `tools` | `tools[].function` | `tools[]` (`type: function`) |
| `OutputFormat::Json` | depends on `JsonMode` (below) | schema in `instructions`; `text.format` too with `.native_structured_output(true)` |
| `max_output_tokens` | `max_tokens` | not sent (undocumented on the Codex backend) |
| images (`Part::Image`, user messages) | `image_url` content parts, only with `.vision(true)` (default for `openai()`); otherwise the call fails | `input_image` (`detail: auto`) |
| `cache_key` | `prompt_cache_key` only with `.send_cache_key(true)` (default for `openai()`) | `prompt_cache_key` |

Replies come back as one assistant message with text and/or tool calls, a
`FinishReason` (`Stop`, `ToolCalls`, `Length`, `ContentFilter`), and usage
when the provider reports it. `Usage` prints as
`in 2200 (cached 960, 44%) · out 130 (reasoning 30)`:

| `Usage` field | OpenAI-compatible | Codex (Responses API) |
|---|---|---|
| `input_tokens` | `prompt_tokens` | `input_tokens` |
| `cached_input_tokens` (part of input) | `prompt_tokens_details.cached_tokens`, DeepSeek `prompt_cache_hit_tokens`, or `cached_tokens` | `input_tokens_details.cached_tokens` |
| `output_tokens` | `completion_tokens` | `output_tokens` |
| `reasoning_tokens` (part of output) | `completion_tokens_details.reasoning_tokens` | `output_tokens_details.reasoning_tokens` |

See DESIGN.md, "Prompt caching", for how to order prompts so the cache hits. On Codex, tool calls are collected from the
stream (`response.output_item.done`) and from the final response.

`Llm::chat_streaming` (and `complete_streaming`, and `Toolbox::run` through
`LoopEvent::Text`) delivers text as it is generated:

| Provider | Streaming |
|---|---|
| `CodexLlm` | always streams; each `response.output_text.delta` becomes a delta |
| `OpenAiCompatLlm` | `chat` sends `stream: false`; `chat_streaming` sends `stream: true` with `stream_options.include_usage` and reassembles text, tool-call fragments, finish reason, and usage |
| `FakeLlm` | word by word, for tests and demos |
| any other `LlmProvider` | the default: the whole text as one delta once the reply is complete |

JSON output on OpenAI-compatible servers, since support differs:

| `JsonMode` | Sends | Default for |
|---|---|---|
| `Schema` | `response_format: json_schema` | `OpenAiCompatLlm::openai` |
| `Object` | `response_format: json_object` + schema in the system prompt | `OpenAiCompatLlm::deepseek` |
| `Instructions` | schema in the system prompt only | `OpenAiCompatLlm::new` (any server) |

Whatever the mode, `Llm::complete_as` validates by deserializing and retries
once with the error shown to the model.

## Errors

Provider failures are `LlmError` values. They are returned to the agent body,
**not** raised as `RunError`, so an agent decides for itself what an HTTP 401,
a rate limit, or `Insufficient Balance` means. Messages include the HTTP
status, the provider's error body, and a hint where one helps.

## Current limits

- Tool calls within one model turn run sequentially in `Toolbox::run`.
- A local server that rejects `stream_options` fails on `chat_streaming`;
  plain `chat` still works.
- No retries or backoff.
- Tests use a local mock server; they check the exact request each provider
  sends and how it parses replies. They cannot exercise the live services from
  CI.
