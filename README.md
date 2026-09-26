# worldfn

A minimal, Bevy-inspired typed function runtime for AI agents.

An agent is an ordinary `async fn`. **Its parameter types are the declarative
description of the context and capabilities it needs**; the world resolves them
by type before calling it.

```rust
async fn researcher(
    llm: Llm,
    web: Tool<WebSearch>,
    memory: Context<RelevantMemory>,
) -> Answer { /* ... */ }

let answer = world.run(researcher).await?;
```

Because the signature is the spec, you can inspect an agent's needs before
building a world, and an under-provisioned world reports *every* missing piece
at once, without calling the function:

```text
$ cargo run --example researcher
researcher needs:
  - Llm
  - Tool<web_search>
  - Context<worldfn::context::RelevantMemory> (needs Res<worldfn::context::MemoryStore>, Res<worldfn::context::Task>)

agent `researcher::researcher` cannot run; missing:
  - Llm
  - Tool<web_search>
  - Context<worldfn::context::RelevantMemory> (needs Res<worldfn::context::MemoryStore>, Res<worldfn::context::Task>)
```

Milestone 1 status: no real LLM provider, no proc macros, zero runtime
dependencies (`tokio` is used only by tests and the example).

## Architecture: the Bevy mapping

| Bevy                  | worldfn                          | Role |
|-----------------------|----------------------------------|------|
| `World`               | `AgentWorld`                     | Type-keyed resource map. LLMs and tools are stored as ordinary resources. |
| `SystemParam`         | `AgentParam`                     | `describe()` (static requirements) + `fetch(&world)` (resolve, or return all unmet requirements). |
| `SystemParamFunction` | `AgentFunction<Marker>`          | Implemented by macro for every `Fn(P0..Pn) -> Fut` with `Pi: AgentParam`. |
| `System`              | `Agent`                          | Object-safe runnable: `name`, `requirements`, `validate`, `start`. |
| `IntoSystem`          | `IntoAgent<Marker>`              | Functions and existing agents both convert. |
| `FunctionSystem`      | `FunctionAgent<Marker, F>`       | Glue: resolves `F::Param`, calls `F`, boxes the future. |
| `Res<T>`              | `Res<T>`                         | Owned `Arc<T>` handle (see below). |
| —                     | `Llm`, `Tool<T: ToolSpec>`       | Capability params. The spec (`WebSearch`) is separate from the handler (real or `FakeTool`). |
| —                     | `Context<C: ContextSource>`      | *Derived* context. `C::Deps` is itself an `AgentParam`, so contexts compose and their needs nest in `Requirement::Context`. |
| —                     | `Option<P>`, tuples              | Optional capabilities; tuple nesting. |

Files: `src/param.rs`, `src/function.rs`, `src/agent.rs`, `src/world.rs`,
`src/llm.rs`, `src/tool.rs`, `src/context.rs`. Tests are in `tests/runtime.rs`.

## Design compromises forced by Rust

### 1. Parameters are owned, not borrowed (the big one)

Bevy's `SystemParam` has `type Item<'w, 's>` so `Res<'w, T>` borrows the world
for the duration of a (synchronous) call. The natural async translation would be:

```rust
async fn researcher(llm: &Llm, memory: Res<'_, Memory>) -> Answer
```

That desugars to `for<'w> Fn(&'w Llm, Res<'w, Memory>) -> impl Future + 'w`.
The future's type **depends on the higher-ranked lifetime `'w`**, so there is
no single `Fut` type to put in a bound like `F: Fn(P) -> Fut`, which is what
the blanket `AgentFunction` impls need. Workarounds all cost something:

- `AsyncFn` / async closures (Rust 1.85) can express the higher-ranked borrow,
  but you cannot yet bound the returned future as `Send` without return-type
  notation (unstable), and combining `AsyncFn` with the GAT projection
  `P::Item<'w>` in blanket impls runs into inference and coherence problems.
- Boxing through a helper trait (`for<'w> Fn(..) -> BoxFuture<'w, _>`) works
  for closures but not for plain `async fn` items, which breaks the headline
  API.

So `AgentParam: 'static`, and params are cheap `Arc` handles cloned out of the
world. Consequences:

- Plain `async fn` works with no annotations or macros.
- `world.run(..)` resolves parameters **eagerly** and returns a
  `Send + 'static` future. It does not borrow the world, so it can be
  `tokio::spawn`ed, and the world can even be dropped while it runs (tested).
- No `ResMut`. Bevy's scheduler can hand out `&mut T` because it knows which
  systems run when. Agents run concurrently and hold params across `.await`s,
  so shared mutable state has to use interior mutability (`Mutex` in `T`).
- Bevy's per-system `State` (`init_state` / `get_param(state, ..)`) is dropped;
  it exists mainly to cache query and archetype data, which we don't have.
  It can come back if params need per-agent caches.

### 2. `async fn` in traits is not object-safe

`LlmProvider` and `ToolHandler` are stored as `Arc<dyn ..>`, so they return
`BoxFuture` instead of being `async fn`. The same applies to `Agent::start`,
which boxes one future per run so heterogeneous agents can live in a
`Vec<Box<dyn Agent<Output = O>>>`. `AgentFunction` itself stays unboxed: the
concrete future type is nameable as an associated type because it is a generic
parameter of the per-arity impl.

### 3. `Send` everywhere

The returned future is required to be `Send` so agents are spawnable on
multi-threaded runtimes. An agent that holds a non-`Send` value (for example
an `Rc` or a `MutexGuard`) across an `.await` will not implement
`AgentFunction`. The compiler error then shows up at `world.run(..)` as an
unsatisfied `IntoAgent` bound, not at the `.await` that caused it. A
`!Send` local-executor variant would need a parallel set of traits.

### 4. Arity and the `Marker` parameter

Rust has no variadic generics, so `AgentParam` for tuples and `AgentFunction`
for `Fn(P0..Pn)` are generated by an `all_tuples!` macro for 0..=12 params.
For more, nest tuples: `(a, b): (Res<A>, Res<B>)` is one param (tested).

Blanket impls for `Fn(A)` and `Fn(A, B)` would overlap under coherence
(nothing stops a type from implementing both), so, as in Bevy, each arity's
impl is for a distinct trait `AgentFunction<fn(P0..Pn) -> Fut>`. `IntoAgent`
uses marker types the same way, so functions and prebuilt agents can both be
passed to `world.run`. Users never write the marker, but it shows up in
compiler errors.

### 5. Parameter resolution is synchronous

`AgentParam::fetch` is a plain `fn`. That keeps resolution atomic and lets the
run future be `'static`. It also means a `ContextSource` that needs I/O
(an embedding search, say) can't `await` during resolution. Options for
milestone 2: an async `fetch` whose future borrows the world (reintroducing
the lifetime from point 1 on the resolution phase only), or a context that
returns a lazy handle the agent awaits itself.

### 6. Misc

- Resources are keyed by `TypeId`, so there is one `Llm` per world. Multiple
  models would need marker-typed handles like `Llm<Fast>` / `Llm<Smart>`,
  following the same pattern as `Tool<T>`.
- `Requirement` names come from `std::any::type_name`, which is
  human-readable but not guaranteed stable.
- `RelevantMemory` is naive word overlap between `MemoryStore` and `Task`,
  a stand-in for retrieval that shows how derived context composes.

## Running

```sh
cargo test
cargo run --example researcher
```
