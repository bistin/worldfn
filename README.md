# worldfn

> **Function signature declares the world it needs.**

A minimal, Bevy-inspired typed function runtime for AI agents. You write an
ordinary Rust `async fn`. Its parameter types declare the dependencies,
tools, and context it needs, and an `AgentWorld` prepares, validates, and runs
it. There is no per-function adapter and no proc macro.

```rust
use worldfn::prelude::*;

async fn researcher(
    llm: Llm,
    web: Tool<WebSearch>,
    memory: Context<RelevantMemory>,
) -> Answer {
    let hits = web.call("bevy system params".into()).await?;
    // ...
}

let mut world = AgentWorld::new();
world
    .provide_llm(FakeLlm::with_answer("A concise answer"))?
    .provide_tool::<WebSearch>(FakeTool::with_response(search_results))?
    .provide_context(RelevantMemory::new(["user likes Bevy"]))?;

let answer = world.run(researcher).await?;
```

## What you get from the signature

`cargo run --example researcher` prints:

```text
researcher
├── requires Llm
├── can call Tool<web_search>
└── requires Context<RelevantMemory>

Cannot prepare researcher:
  ✓ Llm
  ✗ Tool<web_search>: no provider registered
  ✗ Context<RelevantMemory>: no provider registered
```

The first block comes from `researcher.into_agent().meta()` and needs no
world. The second is the error from a world that has only an LLM. Missing
dependencies are all reported at once, before the function body runs.

## API at a glance

| Item | Purpose |
|---|---|
| `AgentWorld::provide`, `provide_llm`, `provide_tool::<T>`, `provide_context` | Bind a value to a logical requirement. Rejects duplicates. |
| `AgentWorld::replace` | Rebind explicitly. Invalidates agents prepared earlier. |
| `AgentWorld::run(f)` | One-shot: prepare, then run. |
| `AgentWorld::prepare(f)` + `run_prepared(&mut agent)` | Initialize parameter state once, then run many times. |
| `Llm`, `Tool<T: ToolSpec>`, `Context<T>`, `Res<T>`, `Option<P>`, tuples | Built-in parameters. |
| `AgentParam` | Implement it to add a parameter kind: `describe` / `init` / `resolve`. |
| `FakeLlm`, `FakeTool<T>` | Deterministic fakes that record calls, for tests and examples. |
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

## Limitations (M1)

- **Parameters are owned handles, not borrows from the world.** A borrowed
  parameter would make the future's type depend on a lifetime, which the
  `FnMut(P) -> Fut` bound cannot express. See `DESIGN.md`. As a result there
  is no `ResMut` or `Write<T>`; shared mutation needs interior mutability.
- **Context is a pre-materialized snapshot.** Nothing is retrieved, ranked,
  or budgeted per task yet.
- **Resolution is synchronous.** The agent body is async.
- **Arity is 0–8.** Group parameters into tuples for more.
- **Futures must be `Send`.** An agent that holds a `!Send` value across an
  `.await` shows up as an unsatisfied `IntoAgent` bound at `world.run`.
- **Some dynamic dispatch.** LLM and tool providers are type-erased
  (`Arc<dyn …>`, one boxed future per call). `Agent::start` boxes one future
  per run.
- **There is one `Llm` per world.** Several models would need marker-typed
  handles such as `Llm<Fast>`.
- **No real LLM provider, capability enforcement, graphs, or scheduler.**

Details: [`DESIGN.md`](DESIGN.md). Background and roadmap:
[`docs/design-discussion.md`](docs/design-discussion.md).

## Development

```sh
cargo test
cargo clippy --all-targets
cargo run --example researcher
```

MSRV is 1.85 (edition 2024). The library has no dependencies; `tokio` is a
dev-dependency.
