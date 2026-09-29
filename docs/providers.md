# LLM providers

Agents only ever see `Llm`. The provider behind it is chosen when the world is
built, so switching between a fake, a subscription, and an API key never
touches agent code.

| Cargo feature | Type | Auth | Endpoint |
|---|---|---|---|
| none | `FakeLlm` | none | none; deterministic, for tests |
| `codex` | `providers::CodexLlm` | ChatGPT subscription, via the official Codex CLI's saved login | `chatgpt.com/backend-api/codex/responses` (unofficial) |
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

How it works:

1. Log in with the **official** Codex CLI: `codex login`, and choose ChatGPT.
   It saves tokens to `$CODEX_HOME/auth.json`, or `~/.codex/auth.json` by
   default. worldfn implements no login flow of its own and never handles
   your password.
2. `CodexLlm::from_codex_home(model)` reads that file. It re-reads the file on
   every call, so when the Codex CLI refreshes the token, the next call picks it
   up.
3. worldfn **never refreshes tokens itself**. When the saved token expires, the
   next call fails with a message telling you to run `codex` (which refreshes
   the login) or `codex login` again.
4. Requests follow the open-source Codex CLI and pi's `openai-codex` provider:
   a Responses-API body with `store: false` and `stream: true`, the
   `chatgpt-account-id` and `OpenAI-Beta: responses=experimental` headers, and
   a server-sent-event reply. The client identifies itself honestly as
   `originator: worldfn`.

It will not work if:

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

## Errors

Provider failures are `LlmError` values. They are returned to the agent body,
**not** raised as `RunError`, so an agent decides for itself what an HTTP 401,
a rate limit, or `Insufficient Balance` means. Messages include the HTTP
status, the provider's error body, and a hint where one helps.

## Current limits

- One prompt in, text out. There are no chat histories, tool calls, images, or
  streaming to the agent yet.
- No retries or backoff.
- Tests use a local mock server; they check the exact request each provider
  sends and how it parses replies. They cannot exercise the live services from
  CI.
