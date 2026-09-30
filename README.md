# worldfn

> **Function signature declares the world it needs.**

A minimal, Bevy-inspired typed function runtime for AI agents. You write an
ordinary Rust `async fn`. Its parameter types declare the dependencies,
tools, inputs, and context it needs, and an `AgentWorld` prepares, validates,
and runs it. There is no per-function adapter and no proc macro.

Context is part of that declaration. `Context<RelevantMemory<5>>` means
"before this runs, retrieve up to 5 memories relevant to this invocation's
task", and the runtime does that retrieval for every call.

```rust
use worldfn::prelude::*;

async fn researcher(
    task: Input<Task>,
    llm: Llm,
    web: Tool<WebSearch>,
    memory: Context<RelevantMemory<2>>,
) -> Answer {
    let hits = web.call(task.0.clone()).await?;
    // ...
}

let mut world = AgentWorld::new();
world
    .provide_llm(FakeLlm::with_answer("A concise answer"))?
    .provide_tool::<WebSearch>(FakeTool::with_response(search_results))?
    .provide_memory(FakeMemory::new(["user likes Bevy", "user deploys with kubernetes"]))?;

let answer = world
    .run_with(researcher, Scope::of(Task::new("How do Bevy system params work?")))
    .await?;
```

## What you get from the signature

`cargo run --example researcher` prints:

```text
researcher
├── reads Input<Task>
├── requires Llm
├── can call Tool<web_search>
└── requires Context<RelevantMemory<2>>
    ├── requires Memory
    └── reads Input<Task>

Cannot prepare researcher:
  · Input<Task>: checked when started
  ✓ Llm
  ✗ Tool<web_search>: no provider registered
  ✗ Context<RelevantMemory<2>>: needs Memory

task:    How do Bevy system params work?
answer:  (fake) Known about the user: ["user likes Bevy's system params"]
sources: ["https://docs.rs/bevy_ecs"]

task:    Should the Rust agent runtime use kubernetes?
answer:  (fake) Known about the user: ["user is building an agent runtime in Rust", "user deploys with kubernetes"]
sources: ["https://docs.rs/bevy_ecs"]

web_search requests: ["How do Bevy system params work?", "Should the Rust agent runtime use kubernetes?"]
```

The tree comes from `researcher.into_agent().meta()` and needs no world.
The ✓/✗ report is the error from a world that has only an LLM: every
missing binding is reported at once, before the function body runs. The two
runs share one prepared agent, and each gets memory retrieved for its own
task.

## API at a glance

| Item | Purpose |
|---|---|
| `AgentWorld::provide`, `provide_llm`, `provide_tool::<T>`, `provide_memory` | Bind a value to a logical requirement. Rejects duplicates. |
| `AgentWorld::replace` | Rebind explicitly. Invalidates agents prepared earlier. |
| `AgentWorld::run_with(f, scope)` / `run(f)` | One-shot: prepare, then run with this invocation's inputs. |
| `AgentWorld::prepare(f)` + `run_prepared_with(&mut agent, scope)` | Initialize parameter state once, then run many times. |
| `Scope::of(Task::new(..))` | Per-invocation inputs, read by `Input<T>`. |
| `Llm`, `Tool<T: ToolSpec>`, `Memory`, `Res<T>`, `Input<T>`, `Option<P>`, tuples | Built-in parameters. |
| `Context<S: ContextSource>` | Context materialized per invocation. `S::Deps` declares what building it needs; `materialize` may be `async`. |
| `RelevantMemory<N>` | Built-in context: up to `N` entries from `Memory` for the `Input<Task>`. |
| `AgentParam` | Implement it to add a parameter kind: `describe` / `init` / `resolve`. |
| `FakeLlm`, `FakeTool<T>`, `FakeMemory` | Deterministic fakes that record calls, for tests and examples. |
| `RunError` | Runtime failures only. The function's own `Result` stays nested: `.await??`. |

Bevy mapping:

| Bevy | worldfn |
|---|---|
| `World` | `AgentWorld` |
| `SystemParam` | `AgentParam` |
| `SystemParamFunction` | `AgentFunction` |
| `System` | `Agent` |
| `IntoSystem` | `IntoAgent` |
| `FunctionSystem` | `FunctionAgent` |
| `SystemState` | `prepare` |

## Talking to models

`Llm` takes a provider-neutral `ChatRequest` (system prompt, messages of text /
tool-call / tool-result parts, tool definitions, output format) and returns one
assistant `ChatResponse` (message, finish reason, token usage). The shapes follow
what current SDKs converge on (Vercel AI SDK, pi-ai, rig); worldfn owns these
types so the core has no provider dependency.

```rust
// Plain text.
let text = llm.complete("Summarize this").await?;

// Typed reply: the JSON Schema comes from the type, the reply is validated by
// deserializing into it, and invalid output gets one corrective retry.
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Triage { category: Category, priority: Priority, reply: String }

let triage: Triage = llm
    .complete_as(ChatRequest::new().system("You triage tickets.").user(ticket_text))
    .await?;
```

Typed replies are the `structured` feature (on by default; it adds `serde`,
`serde_json` and `schemars`). Providers map the request to their wire format,
including tool definitions, tool calls and results, and native JSON output
where it exists. See `docs/providers.md`.

### Tool calling

The tools a model may call are part of the signature, and that tuple is the
model's whole tool surface:

```rust
async fn analyst(task: Input<Task>, llm: Llm, tools: Toolbox<(GrowthRate, Quote)>)
    -> Result<ToolRun, ToolLoopError>
{
    // An explicit, bounded loop: call the model, run the tools it asks for,
    // send the results back, until it answers or 4 model calls are used.
    tools.run(&llm, ChatRequest::new().user(task.0.clone()), 4, |event| { /* progress */ }).await
}
```

- Each tool is an ordinary `ToolSpec`. The schema the model sees is generated
  from `Request`, and `DESCRIPTION` tells it when to use the tool.
- The model's arguments are deserialized into `Request` before the handler
  runs.
- Invalid arguments, tool errors, and calls to tools **outside the tuple** come
  back to the model as error results. The handler is never reached.
- Agents that want their own control flow use `tools.definitions()` and
  `tools.dispatch(&call)` directly.

`cargo run --example mentor_tools` shows numbers computed by a tool and
explained by the model.

## Skills

worldfn reads standard [Agent Skills](https://agentskills.io/specification)
(`<name>/SKILL.md` with `name` / `description` frontmatter), so existing skill
folders work as they are. As with everything else, the signature says how an
agent uses them:

```rust
async fn support(
    question: Input<Question>,
    skills: Context<RelevantSkills<2, Question>>,  // up to 2 skills picked for this question
    catalog: Context<SkillCatalog>,               // or: names + descriptions of all skills
    llm: Llm,
) -> Result<String, LlmError> {
    llm.complete(format!("{}\n\n{}", skills.prompt(), question.0)).await
}

world.provide_skills(SkillLibrary::from_dir("skills")?)?;
```

Selection is currently done by the runtime, before the body runs, using word
overlap that ignores words common to most skills. That is a deterministic
stand-in for semantic matching. Once models can call tools, loading a skill can
also become a tool the model calls. Bundled files are listed for the model but
never read or executed. Skill text becomes model instructions, so only load
skills you trust.

## Sessions and account memory

Conversation history is per session; long-term memory is per account. The
caller authenticates and puts a `Principal { account, session }` in the scope,
and memory parameters bind to it, so **an agent can only reach the caller's own
memory**; no parameter lets it name another account.

```rust
async fn assistant(
    question: Input<Question>,
    history: Context<Conversation<10>>,     // read: last 10 turns of this session
    recall: Context<Recall<3, Question>>,   // read: 3 memories of this account relevant to the question
    log: SessionLog,                        // write: append to this session
    memory: AccountMemory,                  // write: remember for this account
    llm: Llm,
) -> ...
```

An agent that declares only the `Context` types is read-only. Storage sits
behind two traits (`store::SessionStore`, `store::AccountMemoryStore`) and
takes an account id on every call. `store::Cached<S, C>` puts any `store::Cache`
in front of a store, e.g. Postgres as the source of truth with Redis as a
cache. Writes go to the store first; reads survive a cache outage; and
`forget_account` purges the cache or fails. `store::conformance` is the test
suite every backend and cache combination must pass; the in-memory ones do.

## Web frameworks

The core knows no web framework. It meets them at three framework-neutral
points, and each framework gets a thin adapter crate:

| Core (`worldfn`) | Adapter (`worldfn-axum`) |
|---|---|
| `Scope`: per-invocation inputs | your handler builds it from extractors |
| `Emit<E>` parameter + `EventStream<E>` + `SseEvent` | `worldfn_axum::sse(events)` → `axum::response::Sse` |
| `RunError::http_status()`: 400 caller / 502 upstream / 500 setup | `AgentError` implements `IntoResponse` |

```rust
async fn support_chat(question: Input<Question>, llm: Llm, kb: Context<Kb>, events: Emit<ChatEvent>)
    -> Result<String, LlmError> { /* no web types in here */ }

async fn chat(State(world): State<Arc<AgentWorld>>, Query(q): Query<ChatQuery>) -> Response {
    let (emitter, events) = emit::channel::<ChatEvent>();
    tokio::spawn(world.run_with(support_chat, Scope::of(Question(q.q)).with(emitter)));
    worldfn_axum::sse(events).into_response()
}
```

`cargo run -p worldfn-axum --example chat_server` serves a small chat page on
`http://127.0.0.1:3000` that streams progress and the answer over SSE. The
server `prepare`s its agent at startup, so a missing dependency stops it before
it takes traffic.

## Examples

| Example | Shows | Needs |
|---|---|---|
| `researcher` | Requirement tree, ✓/✗ diagnostics, per-task context | nothing |
| `triage` | Support-ticket triage: typed `enum` output, retrieval of similar past tickets, malformed model output as a domain error, concurrent runs of one prepared agent. `cargo test --example triage` tests the same agent with fakes. | nothing (fake LLM); optionally a real provider |
| `live` | A small assistant against a real model | a provider feature and credentials |
| `mentor_tools` | Tool calling: the model calls `growth_rate`, code computes the number, the model explains it | nothing (scripted fake); optionally a real provider |
| `worldfn-axum` `chat_server` | Chat page: question in, progress + answer streamed over SSE; skills; per-session history and per-account memory ("remember …") | nothing (fake LLM); optionally a real provider |

Examples that accept a real model read `WORLDFN_PROVIDER` / `WORLDFN_MODEL`
(see below) and otherwise fall back to a fake.

## Real models

Agents only see `Llm`, so choosing a model is a world-building decision.
Providers are opt-in cargo features:

| Feature | Provider |
|---|---|
| `openai-compat` | `OpenAiCompatLlm`: API-key access to DeepSeek, OpenAI, or any compatible server |
| `codex` | `CodexLlm`: your ChatGPT subscription through the official Codex CLI's saved login. **Unofficial; personal experiments only.** |
| `codex-login` | Adds worldfn's own ChatGPT login (`worldfn login codex`, browser or device code) with automatic token refresh, so the Codex CLI is not needed. Same caveats. |

```sh
WORLDFN_PROVIDER=deepseek WORLDFN_MODEL=deepseek-v4-flash DEEPSEEK_API_KEY=sk-... \
  cargo run --example live --features openai-compat -- "your question"

# ChatGPT plan: sign in once, then use it like any provider
cargo run --features codex-login --bin worldfn -- login codex   # add --device without a browser
WORLDFN_PROVIDER=codex WORLDFN_MODEL=gpt-5.5 \
  cargo run --example live --features codex-login -- "your question"
```

Setup, caveats, and the Codex terms risk are in
[`docs/providers.md`](docs/providers.md).

## Limitations

- **Parameters are owned handles, not borrows from the world.** A borrowed
  parameter would make the future's type depend on a lifetime, which the
  `FnMut(P) -> Fut` bound cannot express. See `DESIGN.md`. As a result there
  is no `ResMut` or `Write<T>`; shared mutation needs interior mutability.
- **Retrieval is faked.** `FakeMemory` ranks by word overlap. There are no
  token budgets, context caching, or provenance yet, and `N` counts entries,
  not tokens.
- **Inputs are checked at start, not at `prepare`.** They vary per
  invocation. A missing input still fails before the body runs.
- **Arity is 0–8.** Group parameters into tuples for more.
- **Futures must be `Send`.** An agent that holds a `!Send` value across an
  `.await` shows up as an unsatisfied `IntoAgent` bound at `world.run`.
- **Some dynamic dispatch and boxing.** Providers are type-erased
  (`Arc<dyn …>`, one boxed future per call). Each run boxes its future, plus
  one future per tuple, `Option` or `Context` parameter. Tuple elements resolve
  sequentially. No benchmarks have been run yet.
- **There is one `Llm` per world.** Several models would need marker-typed
  handles such as `Llm<Fast>`.
- **Providers are minimal.** One prompt in, text out. There are no tool calls, chat histories, retries, or streaming to the agent.
- **No capability enforcement, graphs, or scheduler.**

Details: [`DESIGN.md`](DESIGN.md). Background and roadmap:
[`docs/design-discussion.md`](docs/design-discussion.md).
Reference app API (investment mentor, first domain):
[`docs/api-spec.md`](docs/api-spec.md).

## Development

```sh
cargo test
cargo clippy --all-targets
cargo run --example researcher
```

MSRV is 1.85 (edition 2024). The core has no dependencies with
`--no-default-features`; the default `structured` feature adds serde and
schemars. The
`codex` / `openai-compat` features add `reqwest` and `serde_json`
(`codex-login` also `sha2`, `getrandom`, and `tokio`) and are tested on
stable 1.94. `tokio` is a dev-dependency.

```sh
cargo test --workspace --all-features   # core, providers (mock server), axum adapter
```
