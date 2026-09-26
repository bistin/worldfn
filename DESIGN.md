# worldfn M1 design note

Scope: the choices this prototype makes for async, lifetimes, arity,
ownership, errors, caching, and dispatch, and why. The full design
discussion and roadmap are in [`docs/design-discussion.md`](docs/design-discussion.md).
Section numbers (§) below refer to that document.

## Layering

```text
async fn researcher(llm: Llm, web: Tool<WebSearch>, memory: Context<RelevantMemory>) -> Answer
        │  FnMut(P0..Pn) -> Fut, every Pi: AgentParam            (AgentFunction<Marker>)
        │  into_agent(): builds AgentMeta from describe(); no world touched
        ▼
FunctionAgent { func, meta, state: Option<(Param::State, WorldStamp)> }
        │  initialize(world): Param::init → state, or Diagnostics listing every ✗
        │  start(world):      check stamp → Param::resolve → func.call → box future
        ▼
AgentFuture<Output>   ('static, owns its inputs, spawnable)
```

| Phase | Work | Runs |
|---|---|---|
| compile time | trait selection per arity, monomorphization | once |
| `into_agent` | `AgentParam::describe` → `AgentMeta` | once per agent |
| `initialize` / `prepare` | `AgentParam::init`: binding lookup, validation, state | once per prepared agent |
| `start` / `run_prepared` | stamp check, `AgentParam::resolve`, call | every invocation |

`world.run(f)` is the one-shot path: it does `prepare` + `start` on every
call and caches nothing. `world.prepare(f)` + `world.run_prepared(&mut agent)`
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
handle (`Llm`, `Tool<T>`, `Context<T>`, `Res<T>`). As a result:

- A plain `async fn` works with no attribute or adapter.
- The run future is `Send + 'static`. It borrows neither the world nor the
  agent, so it can be spawned. A test spawns it and drops the world first.
- There is no `ResMut` / `Write<T>`. Shared mutation uses interior mutability
  inside `T`. Real write semantics (short transactions, command buffers) are
  M2 (§15.2).
- Parameter resolution is synchronous. Async, task-aware context
  materialization would need either a resolve future that borrows the world
  or lazy handles. That is M2 (§9). M1's `Context<T>` is an already
  materialized snapshot.

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
  `downcast`. `Llm`, `Tool<T>` and `Context<T>` are stored under the
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
| `LlmProvider`, `ToolHandler<T>` | `Arc<dyn …>` + `BoxFuture` per call | `Llm` in a signature must not name a backend; `async fn` in traits is not dyn-compatible |
| binding lookup | `HashMap<TypeId>` + downcast | at `init` only; `resolve` clones the cached handle |

Nothing here claims zero overhead. No benchmark was run.

## Deliberately not in M1

The following are left for later (§17–18):

- Real LLM providers.
- Task-aware or async context retrieval, and token budgets.
- `Input<T>` / invocation scope.
- `Read` / `Write` / `Emit` semantics and capability enforcement.
- Graphs and probabilistic transitions.
- Borrowed parameters.
- Proc macros.

`Requirement` names use `std::any::type_name`. They are for diagnostics
only and are not a stable format.
