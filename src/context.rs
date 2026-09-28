use std::any::type_name;
use std::fmt;
use std::future::Future;
use std::ops::Deref;

use crate::{
    AgentParam, AgentWorld, BoxFuture, Input, Memory, ParamError, Requirement, Scope, Task,
};

/// Materializing a context failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextError(pub String);

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "context error: {}", self.0)
    }
}

impl std::error::Error for ContextError {}

/// A kind of context an agent can ask for, materialized fresh for every
/// invocation.
///
/// `Deps` is itself an [`AgentParam`], usually a tuple, so a context declares
/// what it needs the same way an agent does: services from the world,
/// [`Input`]s from the invocation scope, even other contexts. Those needs
/// show up nested under [`Requirement::Context`].
///
/// ```
/// # use worldfn::*;
/// struct Shouted(String);
///
/// impl ContextSource for Shouted {
///     type Deps = Input<Task>;
///     async fn materialize(task: Input<Task>) -> Result<Self, ContextError> {
///         Ok(Shouted(task.0.to_uppercase()))
///     }
/// }
/// ```
pub trait ContextSource: Sized + Send + 'static {
    type Deps: AgentParam;

    fn materialize(deps: Self::Deps) -> impl Future<Output = Result<Self, ContextError>> + Send;
}

/// Parameter: context `S`, materialized for this invocation before the agent
/// body runs.
pub struct Context<S>(S);

impl<S> Context<S> {
    pub fn into_inner(self) -> S {
        self.0
    }
}

impl<S> Deref for Context<S> {
    type Target = S;
    fn deref(&self) -> &S {
        &self.0
    }
}

impl<S: fmt::Debug> fmt::Debug for Context<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Context").field(&self.0).finish()
    }
}

impl<S: ContextSource> AgentParam for Context<S> {
    type State = <S::Deps as AgentParam>::State;
    type Future = BoxFuture<'static, Result<Self, ParamError>>;

    fn describe(out: &mut Vec<Requirement>) {
        let mut needs = Vec::new();
        S::Deps::describe(&mut needs);
        out.push(Requirement::Context {
            type_name: type_name::<S>(),
            needs,
        });
    }

    fn init(world: &AgentWorld) -> Result<Self::State, Vec<Requirement>> {
        S::Deps::init(world).map_err(|needs| {
            vec![Requirement::Context {
                type_name: type_name::<S>(),
                needs,
            }]
        })
    }

    fn resolve(state: &mut Self::State, world: &AgentWorld, scope: &Scope) -> Self::Future {
        let deps = S::Deps::resolve(state, world, scope);
        Box::pin(async move {
            let deps = deps.await?;
            S::materialize(deps)
                .await
                .map(Context)
                .map_err(|e| ParamError::failed(type_name::<Self>(), e.0))
        })
    }
}

/// Context: up to `N` memory entries relevant to the invocation's [`Task`],
/// retrieved from the world's [`Memory`] for every run.
///
/// `N` bounds the entry count, not tokens; a token budget is separate,
/// future work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelevantMemory<const N: usize = 5> {
    pub entries: Vec<String>,
}

impl<const N: usize> ContextSource for RelevantMemory<N> {
    type Deps = (Memory, Input<Task>);

    async fn materialize((memory, task): Self::Deps) -> Result<Self, ContextError> {
        let entries = memory.search(&task.0, N).await?;
        Ok(RelevantMemory { entries })
    }
}
