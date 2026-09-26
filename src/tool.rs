use std::any::type_name;
use std::fmt;
use std::future::{Ready, ready};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::param::unmet;
use crate::{AgentParam, AgentWorld, BoxFuture, ParamError, Requirement, Scope};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolError(pub String);

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tool error: {}", self.0)
    }
}

impl std::error::Error for ToolError {}

/// The typed contract of a logical tool, implemented by marker types such as
/// [`WebSearch`]. Real and fake handlers both implement the same contract, so
/// `Tool<WebSearch>` in a signature never changes when the handler does.
pub trait ToolSpec: Send + Sync + 'static {
    const NAME: &'static str;
    type Request: Send + 'static;
    type Response: Send + 'static;
}

/// An implementation of tool `T`. Boxed future because handlers are
/// type-erased behind `Tool<T>` (see [`LlmProvider`](crate::LlmProvider)).
pub trait ToolHandler<T: ToolSpec>: Send + Sync + 'static {
    fn call(&self, request: T::Request) -> BoxFuture<'_, Result<T::Response, ToolError>>;
}

/// Parameter: the world's handler for tool `T`.
pub struct Tool<T: ToolSpec> {
    handler: Arc<dyn ToolHandler<T>>,
}

impl<T: ToolSpec> Tool<T> {
    pub fn new(handler: impl ToolHandler<T>) -> Self {
        Self {
            handler: Arc::new(handler),
        }
    }

    pub async fn call(&self, request: T::Request) -> Result<T::Response, ToolError> {
        self.handler.call(request).await
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
    type State = Tool<T>;
    type Future = Ready<Result<Self, ParamError>>;

    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Tool {
            name: T::NAME,
            spec: type_name::<T>(),
        });
    }

    fn init(world: &AgentWorld) -> Result<Self::State, Vec<Requirement>> {
        world
            .resource::<Tool<T>>()
            .cloned()
            .ok_or_else(unmet::<Self>)
    }

    fn resolve(state: &mut Self::State, _world: &AgentWorld, _scope: &Scope) -> Self::Future {
        ready(Ok(state.clone()))
    }
}

type Respond<T> =
    dyn Fn(&<T as ToolSpec>::Request) -> Result<<T as ToolSpec>::Response, ToolError> + Send + Sync;

/// A deterministic handler for tool `T` that records every typed request.
/// Clones share the record, so keep a clone after `provide_tool`.
pub struct FakeTool<T: ToolSpec> {
    respond: Arc<Respond<T>>,
    requests: Arc<Mutex<Vec<T::Request>>>,
}

impl<T: ToolSpec> FakeTool<T> {
    /// Compute each response from the request.
    pub fn new(
        respond: impl Fn(&T::Request) -> Result<T::Response, ToolError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            respond: Arc::new(respond),
            requests: Arc::default(),
        }
    }

    /// Always return a clone of `response`.
    pub fn with_response(response: T::Response) -> Self
    where
        T::Response: Clone + Sync,
    {
        Self::new(move |_| Ok(response.clone()))
    }

    /// Always fail with `message`.
    pub fn failing(message: impl Into<String>) -> Self {
        let message = message.into();
        Self::new(move |_| Err(ToolError(message.clone())))
    }

    /// Every request received so far, in order.
    pub fn requests(&self) -> Vec<T::Request>
    where
        T::Request: Clone,
    {
        self.lock().clone()
    }

    pub fn calls(&self) -> usize {
        self.lock().len()
    }

    fn lock(&self) -> MutexGuard<'_, Vec<T::Request>> {
        self.requests.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl<T: ToolSpec> Clone for FakeTool<T> {
    fn clone(&self) -> Self {
        Self {
            respond: self.respond.clone(),
            requests: self.requests.clone(),
        }
    }
}

impl<T: ToolSpec> ToolHandler<T> for FakeTool<T> {
    fn call(&self, request: T::Request) -> BoxFuture<'_, Result<T::Response, ToolError>> {
        let response = (self.respond)(&request);
        self.lock().push(request);
        Box::pin(std::future::ready(response))
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
    type Request = String;
    type Response = Vec<SearchHit>;
}
