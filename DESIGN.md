# worldfn design note

Covers M1 and the first M2 slice (invocation inputs and per-invocation
context). Scope: the choices this prototype makes for async, lifetimes, arity,
ownership, errors, caching, and dispatch, and why. The full design
discussion and roadmap are in [`docs/design-discussion.md`](docs/design-discussion.md).
Section numbers (§) below refer to that document.

## Layering

```text
async fn researcher(llm: Llm, web: Tool<WebSearch>, memory: Context<RelevantMemory>) -> Answer
        │  FnMut(P0..Pn) -> Fut, every Pi: AgentParam            (AgentFunction<Marker>)
        │  into_agent(): builds AgentMeta from describe(); no world touched
        ▼
FunctionAgent { func: Arc<Mutex<F>>, meta, state: Option<(Param::State, WorldStamp)> }
        │  initialize(world):   Param::init → state, or Diagnostics listing every ✗
        │  start(world, scope): check stamp → Param::resolve (sync part) ──┐
        ▼                                                                   │
AgentFuture<Output>   ('static, owns its inputs, spawnable)                │
        │  await resolution: context materialization, retrieval… ◄───────┘
        │  lock func just long enough to call it
        ▼
        await body
```

| Phase | Work | Runs |
|---|---|---|
| compile time | trait selection per arity, monomorphization | once |
| `into_agent` | `AgentParam::describe` → `AgentMeta` | once per agent |
| `initialize` / `prepare` | `AgentParam::init`: binding lookup, validation, state | once per prepared agent |
| `start` / `run_prepared_with` | stamp check, sync part of `AgentParam::resolve` | every invocation |
| awaiting the run future | async part of `resolve` (e.g. `ContextSource::materialize`), then the body | every invocation |

`world.run(f)` is the one-shot path: it does `prepare` + `start` on every
call and caches nothing. `world.prepare(f)` + `world.run_prepared_with(&mut agent, scope)`
is the path that reuses initialization. A test checks that `init` runs once
and `resolve` runs once per invocation.

## Ownership: parameters are owned, not borrowed

Bevy's `SystemParam::Item<'w, 's>` borrows from the world for one
synchronous call. An agent is an `async fn`, and its future outlives the call
and holds parameters across `.await`s. If parameters borrowed the world, the
function would be `for<'w> FnMut(Res<'w, T>) -> impl Future + 'w`: the
future's type depends on a higher-ranked lifetime, so there is no single
`Fut` to put in the `FnMut(P) -> Fut` bound that the per-arity impls rely on.
The alternatives all cost something today:

- `AsyncFnMut` can express the borrow, but a `Send` bound on its future needs
  return-type notation (unstable). Combining it with the `P::Item<'w>` GAT
  projection in blanket impls also runs into inference and coherence
  problems.
- `for<'w> FnMut(..) -> BoxFuture<'w, _>` works for closures but not for
  plain `async fn` items, which breaks the target API.

So `AgentParam: 'static` and `resolve` returns `Self`: an `Arc`-backed
handle (`Llm`, `Tool<T>`, `Memory`, `Res<T>`, `Input<T>`), or a value
built from such handles (`Context<S>`). As a result:

- A plain `async fn` works with no attribute or adapter.
- The run future is `Send + 'static`. It borrows neither the world nor the
  agent, so it can be spawned. A test spawns it and drops the world first.
- There is no `ResMut` / `Write<T>`. Shared mutation uses interior mutability
  inside `T`. Real write semantics (short transactions, command buffers) are
  M2 (§15.2).
- Resolution can still be async; see the next section.

## Async resolution without borrowed parameters (M2 slice)

Context has to be built per invocation from the task (§9), and building it
may await I/O. The obvious approach, an `async fn resolve(&World)`, brings
back exactly the borrowed-future problem above. Instead, `resolve` is split
at the type level:

```rust
fn resolve(state: &mut Self::State, world: &AgentWorld, scope: &Scope) -> Self::Future;
//         └─ sync: may read world/scope, but only to clone owned handles out
type Future: Future<Output = Result<Self, ParamError>> + Send + 'static;
//         └─ async: owns everything, borrows nothing
```

For example, `Context<RelevantMemory<N>>` clones the `Memory` handle and the
`Input<Task>` synchronously. Its future then awaits
`memory.search(&task, N)`. The whole run future stays `'static`, so it can be
spawned, and the world can be mutated or dropped while retrieval is in
flight (tested).

Consequences:

- The function is called *after* asynchronous resolution finishes, inside the
  `'static` future. So `FunctionAgent` keeps it in `Arc<Mutex<F>>`, locked only
  for the synchronous `call`, never across an `.await`. Stateful `FnMut`
  closures keep working. Concurrent runs of one prepared agent serialize
  only on that brief call.
- `Input<T>` values come from a per-invocation `Scope`, not from the world:
  `world.run_with(agent, Scope::of(Task::new(..)))`. This avoids a global
  mutable "current task" (§9). Inputs vary per call, so `prepare` cannot check
  them. A missing input fails at start, before any retrieval and before the
  body runs.
- `ContextSource` declares its needs as `type Deps: AgentParam`. So contexts
  depend on services, inputs, and other contexts the same way agents do, and
  `describe` nests them in the metadata tree. `materialize` is an
  `impl Future + Send` trait method, which implementors write as `async fn`.
- Tuple elements start resolving in order during the sync phase. Their futures
  are then awaited **sequentially**. Joining them concurrently would need a
  per-arity join future or a dependency.
- Only *binding* absence makes `Option<P>` degrade to `None`. A bound
  parameter that fails to resolve, such as a failing retrieval, is still an
  error (§9: only explicitly optional context may degrade).

## Session and account memory

Memory has two scopes with different access patterns, so it has two traits
rather than one generic store:

- `SessionStore` holds conversation turns and is read by recency.
- `AccountMemoryStore` holds long-term entries and is read by relevance.

Every method takes the owning account, and session keys are
`(account, session)`, so a guessed session id from another account reads
nothing.

Agents never see the traits. `SessionLog` and `AccountMemory` are handles
bound during `resolve` to the `Principal` in the invocation scope. There is
deliberately no public parameter wrapping the raw store, because that would
let an agent pass any account id. `Context<Conversation<N>>` and
`Context<Recall<N, Q>>` are read-only snapshots built from those handles, so
the read/write capability shows in the signature. A missing `Principal` is a
caller error (400).

Caching is a decorator, not a backend. `Cached<S, C>` implements both traits
over any store `S` and any `C: Cache` (get / set / delete / delete-prefix with
TTL). A Redis crate therefore only implements `Cache`, and the cache policy is
written and tested once:

- Session reads fill a per-session window.
- Writes go to the store, then invalidate the cached window.
- Cache failures degrade to store reads.
- Account searches bypass the cache.
- `forget_account` must also purge every cached key of the account, or it
  returns an error. A deletion is never reported done while copies may remain.

Cache keys length-prefix ids so they cannot collide across accounts.
`store::conformance` holds the shared behavioral tests; backend crates run it
against their own stores.

## The LLM contract

`LlmProvider` has one method: `chat(ChatRequest) -> ChatResponse`. The types
live in the core (`worldfn::chat`) and are shaped after what current SDKs
converge on:
- a system prompt;
- messages made of text, tool-call and tool-result parts;
- tool definitions;
- an output format;
- in the reply, a finish reason and token usage.

Owning them, rather than adopting a crate's types, keeps the core free of
provider dependencies and lets `FakeLlm` script any reply, including tool
calls, without a network. A broad provider crate (e.g. rig) can still be
wrapped as one `LlmProvider` adapter later.

JSON payloads (tool arguments, schemas) are JSON *text* in these types. That
keeps `--no-default-features` dependency-free. The `structured` feature adds
`Llm::complete_as::<T>()`, which:
- generates the schema from `T` with `schemars`;
- sends it as the output format;
- validates the reply by deserializing into `T`;
- on failure, shows the model its reply and the error and asks again, up to a
  retry limit.

A truncated reply (`FinishReason::Length`) is never parsed. Rules a type
cannot express, such as a non-empty string, stay in the caller's code, as the
triage example shows.

### Tool calling

`Toolbox<(A, B, ..)>` is a parameter like any other. The tuple is the model's
whole tool surface: definitions are generated from each `ToolSpec` (name,
`DESCRIPTION`, and a schema from `Request`), and `dispatch` only reaches
handlers in the tuple. The model's arguments are untrusted input. They are
deserialized into `Request` before any handler runs. Bad arguments, handler
errors, and calls to unlisted tools become error results the model sees, so it
can recover, and they never become panics or runtime errors.

`Toolbox::run` is the loop: call the model, run the requested tools in order,
append calls and results, and repeat until the model answers or `max_steps`
model calls are used (`ToolLoopError::MaxSteps`). It is a method the agent
calls, with a visible bound, not something the runtime does around the agent
(§3). An observer callback sees streamed reply text (`LoopEvent::Text`) and each
call and result, which is how progress reaches an `Emit` stream. Agents that need other control flow, such as
approval before a tool runs or parallel calls, use `definitions` and
`dispatch` themselves.

Tool arguments must be JSON objects, so a non-object `Request` (e.g.
`String`) is wrapped as `{"input": ...}` and unwrapped on dispatch.

### Streaming

`LlmProvider::chat_streaming` takes a `&mut dyn FnMut(ChatDelta)` callback
and returns the same `ChatResponse` as `chat`. The default implementation
calls `chat` and delivers the whole text as one delta, so a provider that
cannot stream still works wherever streaming is used, and callers need no
second code path. Codex streams the Responses-API deltas. OpenAI-compatible
servers get `stream: true`, and their chunks, including tool-call argument
fragments, are rebuilt into the non-streaming response shape and parsed by
the same code.

A callback rather than a returned `Stream` keeps the core free of a stream
trait and of an async runtime, matching `Emit`. Getting text to a web client
is one line: forward each delta to an `Emit<E>`, and the framework adapter
turns that into SSE. `ChatDelta` is `#[non_exhaustive]`, so reasoning
summaries or tool-call progress can be added later.

Structured output (`complete_as`) does not stream: partial JSON is not useful
to show, and the reply is only valid once it parses.

### Prompt caching

Hosted providers (OpenAI, the Codex backend, DeepSeek) and local servers
(vLLM, llama.cpp) reuse the computed prefix of a prompt they have seen
recently: cached input is faster and cheaper. worldfn cannot control the
cache, only keep prompts cache-friendly, so the rules are about ordering:

1. **Stable first, variable last.** System prompt, then history as real
   messages (`Conversation::messages`), then one final user message with what
   was retrieved for this question (`Context` output, selected skills, recall)
   and the question. Retrieved context in the system prompt changes the first
   tokens and makes every request a miss.
2. **Append, never rewrite.** `Toolbox::run` only appends calls and results,
   keeps system prompt and tool definitions identical between steps, and a
   test checks that each step's request is a prefix of the next. Serialization
   is deterministic (sorted JSON keys, schemas from the types, tool arguments
   echoed as the model wrote them).
3. **One cache key per conversation.** `ChatRequest::cache_key` becomes
   `prompt_cache_key` on Codex and (opt-in) OpenAI, which routes requests with
   the same prefix to the same cache. `Toolbox::run` sets one per run if the
   caller did not; a chat endpoint should use the session id.
4. **Measure it.** `Usage::cached_input_tokens` (and `reasoning_tokens`) are
   parsed from each provider's own field names; `LoopEvent::ModelResponded`
   reports usage per model call.

Known limits: `Conversation<N>` is a sliding window, so past `N` turns its
oldest message changes each turn and the cached prefix stops at the system
prompt. Dropping old tool outputs or screenshots to save context has the same
effect; do it in large, rare steps rather than a little on every call.

Still missing: images.

## Skills

Agent Skills are progressive disclosure: a catalog (level 1), full
instructions when relevant (level 2), and bundled files (level 3). In worldfn
each level is a parameter type, so the signature records how much of the
library an agent can see:

- `Context<SkillCatalog>` is level 1.
- `Context<RelevantSkills<N, Q>>` is level 2, chosen by the runtime from the
  per-invocation input `Q: AsQuery`.
- Level 3 files are only listed, never read or run.

`SkillLibrary` validates on load: names follow the spec and match their
folders, names are unique, and descriptions are 1–1024 characters without
angle brackets. Invalid skills fail loudly instead of being skipped. The
frontmatter reader is a deliberately small subset of YAML (scalars, quoted
strings, block scalars), which keeps the core free of dependencies; nested
fields such as `metadata:` are ignored, as the spec allows. Model-driven
loading (a `load_skill` tool) comes with tool calling.

## Framework seams

worldfn must stay usable from any web framework (and from none), so the core
depends on no framework and meets them at three points:

- **Inputs:** a framework handler turns request data into a `Scope`. Nothing
  in an agent signature is an HTTP type.
- **Streaming output:** `Emit<E>` is a per-invocation capability like
  `Input<T>`: the caller puts an `Emitter<E>` in the scope and keeps the
  `EventStream<E>`. The channel is implemented in the core with only `std`
  (a mutex-protected queue plus a waker), so it works with any executor.
  `poll_next` is the hook adapters use to implement their framework's stream
  type. It is unbounded, and it ends when the last emitter drops, which
  happens when the run finishes. If the receiver is dropped (the client
  disconnected), `send` returns `false`, so agents can stop early.
- **Errors:** `RunError::http_status` classifies failures without naming a
  framework: a missing scope value is the caller's fault (400), a failing
  parameter backend is upstream (502), and missing bindings or stale agents
  are setup errors (500). `ParamError::kind` carries the distinction.

`SseFrame`/`SseEvent` describe events in wire terms without choosing a
serializer; the event type decides its own `data`. Adapters (`worldfn-axum`
first) are separate crates in the workspace, so the compiler enforces that the
core never grows a framework dependency.

## Arity and the `Marker` parameter

Rust has no variadic generics. `macro_rules! all_tuples` generates
`AgentParam` for tuples and `AgentFunction` for `FnMut(P0..Pn)` with
0 ≤ n ≤ 8, following §15.3. To go further, nest tuples, e.g.
`(a, b): (Res<A>, Res<B>)` counts as one parameter. A test covers 8
parameters with a nested tuple.

Blanket impls for `FnMut(A)` and `FnMut(A, B)` could overlap, so each arity
implements a distinct trait, `AgentFunction<fn(P0..Pn) -> Fut>`. As in
Bevy, the marker also constrains `Fut`, which would otherwise be an
unconstrained impl parameter. `IntoAgent` uses marker types the same way, so
both functions and existing agents convert. The cost: markers appear in
compiler errors, and a bad parameter type surfaces as an unsatisfied
`IntoAgent` bound at `world.run(..)`. A `compile_fail` doctest pins this
down.

`FnMut`, not `Fn`, so that stateful closures work. The returned future does
not borrow the function, so multiple futures from one prepared agent may be
in flight at once. The `&mut` is only held while `call` runs.

## Error layering

```text
world.run(f).await : Result<F::Output, RunError>
```

`RunError` is the runtime layer only:

- `Unresolved(Diagnostics)`: lists every declared requirement with ✓/✗.
- `NotPrepared`, `ForeignWorld`, `Stale`: see the next section.
- `Param`: a resolve-time failure.

The function's own output is untouched. An agent returning
`Result<Answer, AgentError>` yields
`Result<Result<Answer, AgentError>, RunError>`, and callers write `.await??`.
There is no flattening, because overlapping blanket impls special-casing
`Result` would hide semantics (§8.3). Provider failures (`LlmError`,
`ToolError`) are values the function sees; they are never reported as
missing dependencies.

## Bindings, identity, invalidation

- The world is a `TypeId → Arc<dyn Any + Send + Sync>` map with safe
  `downcast`. `Llm`, `Tool<T>` and `Memory` are stored under the
  **logical** type, so `provide_llm(FakeLlm)` binds an `Llm`. A fake is never
  matched by its own `TypeId`.
- One binding per key. `provide*` returns `BindError` on duplicates, and
  `replace` rebinds explicitly.
- Every world gets a process-unique id. Every `replace` bumps a generation
  counter. A prepared agent records `(id, generation)` at `initialize`, and
  `start` refuses to run with `ForeignWorld` or `Stale` if either differs. So
  stale handles are never used silently; the caller re-`initialize`s.

## Dispatch and boxing boundaries

| Boundary | Static or dynamic | Why |
|---|---|---|
| `AgentFunction` (fn → future) | static: concrete `F`, concrete `Fut` | nameable as an associated type of the per-arity impl |
| `Agent::start` | one `Box<dyn Future>` per invocation | keeps `Agent` object-safe for `Box<dyn Agent<Output = O>>` |
| `AgentParam::Future` for tuples, `Option`, `Context` | one `BoxFuture` each per invocation | a tuple's combined future, and a `ContextSource`'s `impl Future`, are not nameable as associated types on stable |
| leaf params (`Llm`, `Tool`, `Res`, `Input`, `Memory`) | `std::future::Ready`, no allocation | nothing to await |
| `FunctionAgent::func` | `Arc<Mutex<F>>`, one uncontended lock per call | the function is called after async resolution |
| `LlmProvider`, `ToolHandler<T>`, `MemoryStore` | `Arc<dyn …>` + `BoxFuture` per call | the signature must not name a backend; `async fn` in traits is not dyn-compatible |
| binding lookup | `HashMap<TypeId>` + downcast | at `init` only; `resolve` clones the cached handle |

Nothing here claims zero overhead. No benchmark was run.

## Deliberately not done yet

The following are left for later (§17–18):

- Real LLM providers and real retrieval. `FakeMemory` uses word overlap as a
  deterministic stand-in.
- Token or cost budgets for context. The `N` in `RelevantMemory<N>` is an
  entry count, not a token limit.
- Context caching and provenance (source IDs, revisions).
- `Read` / `Write` / `Emit` semantics and capability enforcement.
- Graphs and probabilistic transitions.
- Borrowed parameters.
- Proc macros.

`Requirement` names use `std::any::type_name`, which, for example, omits
defaulted const parameters (`RelevantMemory`, but `RelevantMemory<2>`).
They are for diagnostics only and are not a stable format.
