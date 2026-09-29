//! # worldfn
//!
//! **Function signature declares the world it needs.**
//!
//! A minimal, Bevy-inspired typed function runtime for AI agents. An agent is
//! an ordinary `async fn`; its parameter types declare the dependencies,
//! tools, and context it needs, and an [`AgentWorld`] prepares, validates, and
//! runs it.
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
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # block_on(async {
//! let mut world = AgentWorld::new();
//! world
//!     .provide_llm(FakeLlm::with_answer("42"))?
//!     .provide_tool::<WebSearch>(FakeTool::with_response(vec![]))?
//!     .provide_memory(FakeMemory::new(["user prefers tokio for async"]))?;
//!
//! // The context is materialized for this task, before the body runs.
//! let answer = world
//!     .run_with(researcher, Scope::of(Task::new("explain rust async")))
//!     .await?;
//! assert_eq!(answer, Answer("42".into()));
//! # Ok(())
//! # })
//! # }
//! # fn block_on<F: std::future::Future>(f: F) -> F::Output {
//! #     tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(f)
//! # }
//! ```
//!
//! A parameter type that is not an [`AgentParam`] is rejected at compile time:
//!
//! ```compile_fail
//! # use worldfn::prelude::*;
//! async fn bad(prompt: String) {}
//! let _ = AgentWorld::new().run(bad);
//! ```
//!
//! | Bevy                  | worldfn            |
//! |-----------------------|--------------------|
//! | `World`               | [`AgentWorld`]     |
//! | `SystemParam`         | [`AgentParam`]     |
//! | `SystemParamFunction` | [`AgentFunction`]  |
//! | `System`              | [`Agent`]          |
//! | `IntoSystem`          | [`IntoAgent`]      |
//! | `FunctionSystem`      | [`FunctionAgent`]  |
//! | `SystemState`         | [`AgentWorld::prepare`] |
//!
//! See `DESIGN.md` for the compromises forced by Rust's async, lifetime, and
//! arity rules.

mod agent;
pub mod chat;
mod context;
pub mod emit;
mod error;
mod function;
mod input;
mod llm;
mod memory;
mod param;
#[cfg(feature = "http")]
pub mod providers;
pub mod scoped;
pub mod skills;
pub mod store;
#[cfg(feature = "structured")]
pub mod structured;
mod tool;
mod world;

pub use agent::{
    Agent, AgentFuture, AgentMeta, FunctionAgent, IntoAgent, IsAgent, IsFunctionAgent,
};
pub use chat::{ChatRequest, ChatResponse, Message, OutputFormat};
pub use context::{Context, ContextError, ContextSource, RelevantMemory};
pub use emit::{Emit, EventStream, SseEvent, SseFrame};
pub use error::{BindError, Check, Diagnostics, ParamError, ParamErrorKind, RunError};
pub use function::AgentFunction;
pub use input::{Input, Scope, Task};
pub use llm::{FakeLlm, Llm, LlmError, LlmProvider};
pub use memory::{FakeMemory, Memory, MemoryStore};
pub use param::{AgentParam, Requirement, Res};
pub use scoped::{AccountMemory, Conversation, Principal, Recall, SessionLog};
pub use skills::{AsQuery, RelevantSkills, Skill, SkillCatalog, SkillLibrary, Skills};
#[cfg(feature = "structured")]
pub use structured::StructuredError;
pub use tool::{FakeTool, SearchHit, Tool, ToolError, ToolHandler, ToolSpec, WebSearch};
pub use world::AgentWorld;

/// A boxed, `Send` future, used at type-erased provider boundaries.
pub type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

pub mod prelude {
    pub use crate::{AccountMemory, Conversation, Principal, Recall, SessionLog};
    pub use crate::{
        Agent, AgentParam, AgentWorld, Context, ContextError, ContextSource, Emit, FakeLlm,
        FakeMemory, FakeTool, Input, IntoAgent, Llm, Memory, RelevantMemory, RelevantSkills,
        Requirement, Res, RunError, Scope, SkillCatalog, SkillLibrary, Skills, Task, Tool,
        ToolSpec, WebSearch,
    };
}

/// Implements a macro for every tuple arity from the given list down to zero.
///
/// Rust has no variadic generics, so — like Bevy's `all_tuples!` — arity
/// support is generated. `all_tuples!(m; A, B)` expands to
/// `m!(A, B); m!(B); m!();`.
macro_rules! all_tuples {
    ($m:ident;) => { $m!(); };
    ($m:ident; $head:ident $(, $tail:ident)*) => {
        $m!($head $(, $tail)*);
        all_tuples!($m; $($tail),*);
    };
}
pub(crate) use all_tuples;
