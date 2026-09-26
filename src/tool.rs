use std::any::type_name;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{AgentParam, AgentWorld, BoxFuture, Requirement};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolError(pub String);

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tool error: {}", self.0)
    }
}

impl std::error::Error for ToolError {}

/// The typed signature of a tool. Implemented by marker types like
/// [`WebSearch`]; the implementation lives in a [`ToolHandler`].
///
/// Separating the spec from the handler is what lets `Tool<WebSearch>` in a
/// function signature be a pure capability *declaration*, bound to a real or
/// fake implementation by whoever builds the world.
pub trait ToolSpec: Send + Sync + 'static {
    const NAME: &'static str;
    type Input: Send + 'static;
    type Output: Send + 'static;
}

/// An implementation of tool `T`. Boxed future for object safety (see
/// [`LlmProvider`](crate::LlmProvider)).
pub trait ToolHandler<T: ToolSpec>: Send + Sync + 'static {
    fn call(&self, input: T::Input) -> BoxFuture<'_, Result<T::Output, ToolError>>;
}

/// Parameter: access to the world's handler for tool `T`.
pub struct Tool<T: ToolSpec> {
    handler: Arc<dyn ToolHandler<T>>,
}

impl<T: ToolSpec> Tool<T> {
    pub fn new(handler: impl ToolHandler<T>) -> Self {
        Self {
            handler: Arc::new(handler),
        }
    }

    pub async fn call(&self, input: T::Input) -> Result<T::Output, ToolError> {
        self.handler.call(input).await
    }
}

impl<T: ToolSpec> Clone for Tool<T> {
    fn clone(&self) -> Self {
        Self {
            handler: self.handler.clone(),
        }
    }
}

impl<T: ToolSpec> AgentParam for Tool<T> {
    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Tool {
            name: T::NAME,
            spec: type_name::<T>(),
        });
    }

    fn fetch(world: &AgentWorld) -> Result<Self, Vec<Requirement>> {
        world.resource::<Tool<T>>().cloned().ok_or_else(|| {
            let mut missing = Vec::new();
            Self::describe(&mut missing);
            missing
        })
    }
}

type FakeFn<T> =
    dyn Fn(<T as ToolSpec>::Input) -> Result<<T as ToolSpec>::Output, ToolError> + Send + Sync;

/// A synchronous, deterministic tool for tests. Clones share the call count.
pub struct FakeTool<T: ToolSpec> {
    f: Arc<FakeFn<T>>,
    calls: Arc<AtomicUsize>,
}

impl<T: ToolSpec> FakeTool<T> {
    pub fn new(
        f: impl Fn(T::Input) -> Result<T::Output, ToolError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            f: Arc::new(f),
            calls: Arc::default(),
        }
    }

    /// How many times the tool has been called.
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl<T: ToolSpec> Clone for FakeTool<T> {
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            calls: self.calls.clone(),
        }
    }
}

impl<T: ToolSpec> ToolHandler<T> for FakeTool<T> {
    fn call(&self, input: T::Input) -> BoxFuture<'_, Result<T::Output, ToolError>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(std::future::ready((self.f)(input)))
    }
}

/// Built-in tool spec: web search.
pub struct WebSearch;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

impl ToolSpec for WebSearch {
    const NAME: &'static str = "web_search";
    type Input = String;
    type Output = Vec<SearchHit>;
}
