//! # worldfn
//!
//! A minimal, Bevy-inspired typed function runtime for AI agents.
//!
//! An agent is an ordinary `async fn`. Its **parameter types are the
//! declarative description of the capabilities and context it needs**; the
//! [`AgentWorld`] resolves them by type before the function is called.
//!
//! ```
//! use worldfn::prelude::*;
//!
//! #[derive(Debug, PartialEq)]
//! struct Answer(String);
//!
//! async fn researcher(
//!     llm: Llm,
//!     web: Tool<WebSearch>,
//!     memory: Context<RelevantMemory>,
//! ) -> Answer {
//!     let hits = web.call("rust async".into()).await.unwrap();
//!     let prompt = format!("{hits:?} {:?}", memory.entries);
//!     Answer(llm.complete(prompt).await.unwrap())
//! }
//!
//! # tokio_test_block_on(async {
//! let mut world = AgentWorld::new();
//! world
//!     .insert_llm(FakeLlm::scripted(["42"]))
//!     .insert_tool::<WebSearch>(FakeTool::new(|_q| Ok(vec![])))
//!     .insert(MemoryStore::default())
//!     .insert(Task::new("what is rust async?"));
//!
//! let answer = world.run(researcher).await.unwrap();
//! assert_eq!(answer, Answer("42".into()));
//! # });
//! # fn tokio_test_block_on<F: std::future::Future>(f: F) -> F::Output {
//! #     tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(f)
//! # }
//! ```
//!
//! The Bevy mapping:
//!
//! | Bevy                  | worldfn            |
//! |-----------------------|--------------------|
//! | `World`               | [`AgentWorld`]     |
//! | `SystemParam`         | [`AgentParam`]     |
//! | `SystemParamFunction` | [`AgentFunction`]  |
//! | `System`              | [`Agent`]          |
//! | `IntoSystem`          | [`IntoAgent`]      |
//! | `FunctionSystem`      | [`FunctionAgent`]  |
//!
//! See `README.md` for the design compromises forced by Rust's async,
//! lifetime, and arity rules.

mod agent;
mod context;
mod function;
mod llm;
mod param;
mod tool;
mod world;

pub use agent::{Agent, AgentFuture, FunctionAgent, IntoAgent, IsAgent, IsFunctionAgent};
pub use context::{Context, ContextSource, MemoryStore, RelevantMemory, Task};
pub use function::AgentFunction;
pub use llm::{FakeLlm, Llm, LlmError, LlmProvider};
pub use param::{AgentParam, Requirement, Res};
pub use tool::{FakeTool, SearchHit, Tool, ToolError, ToolHandler, ToolSpec, WebSearch};
pub use world::{AgentWorld, ResolveError};

/// A boxed, `Send` future. Used wherever a trait must stay object-safe.
pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

pub mod prelude {
    pub use crate::{
        Agent, AgentParam, AgentWorld, Context, ContextSource, FakeLlm, FakeTool, IntoAgent, Llm,
        MemoryStore, RelevantMemory, Requirement, Res, Task, Tool, ToolSpec, WebSearch,
    };
}

/// Implements a macro for every tuple arity from `$max` down to zero.
///
/// Rust has no variadic generics, so — exactly like Bevy's `all_tuples!` —
/// arity support is generated. `all_tuples!(m; A, B)` expands to
/// `m!(A, B); m!(B); m!();`.
macro_rules! all_tuples {
    ($m:ident;) => { $m!(); };
    ($m:ident; $head:ident $(, $tail:ident)*) => {
        $m!($head $(, $tail)*);
        all_tuples!($m; $($tail),*);
    };
}
pub(crate) use all_tuples;
