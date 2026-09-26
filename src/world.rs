use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::sync::Arc;

use crate::{Agent, IntoAgent, Llm, LlmProvider, Requirement, Tool, ToolHandler, ToolSpec};

/// The container agents resolve their parameters from. Analogue of Bevy's
/// `World`, reduced to a type-keyed resource map.
///
/// LLMs and tools are stored as ordinary resources ([`Llm`] and [`Tool<T>`]
/// are themselves cheap handles), so there is one storage and one lookup
/// path.
#[derive(Default)]
pub struct AgentWorld {
    resources: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl AgentWorld {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) a resource.
    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) -> &mut Self {
        self.resources.insert(TypeId::of::<T>(), Arc::new(value));
        self
    }

    /// Register the LLM provider used by `Llm` parameters.
    pub fn insert_llm(&mut self, provider: impl LlmProvider) -> &mut Self {
        self.insert(Llm::new(provider))
    }

    /// Register the handler used by `Tool<T>` parameters.
    pub fn insert_tool<T: ToolSpec>(&mut self, handler: impl ToolHandler<T>) -> &mut Self {
        self.insert(Tool::<T>::new(handler))
    }

    pub fn contains<T: Send + Sync + 'static>(&self) -> bool {
        self.resources.contains_key(&TypeId::of::<T>())
    }

    pub fn resource<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.resources.get(&TypeId::of::<T>())?.downcast_ref()
    }

    pub fn resource_arc<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.resources
            .get(&TypeId::of::<T>())?
            .clone()
            .downcast()
            .ok()
    }

    /// Resolve an agent's parameters and run it.
    ///
    /// Parameters are resolved **eagerly**, when `run` is called, not when
    /// the future is first polled. The returned future is `Send + 'static`:
    /// it does not borrow the world, so it can be spawned, and the world can
    /// be mutated (or dropped) while it is in flight.
    pub fn run<M, A: IntoAgent<M>>(
        &self,
        agent: A,
    ) -> impl Future<Output = Result<<A::Agent as Agent>::Output, ResolveError>>
    + Send
    + 'static
    + use<M, A> {
        self.run_agent(&agent.into_agent())
    }

    /// Like [`run`](Self::run), for an agent you keep and run repeatedly
    /// (including `dyn Agent` trait objects).
    pub fn run_agent<A: Agent + ?Sized>(
        &self,
        agent: &A,
    ) -> impl Future<Output = Result<A::Output, ResolveError>> + Send + 'static + use<A> {
        let started = agent.start(self);
        async move { Ok(started?.await) }
    }

    /// Check whether this world can satisfy an agent, without running it.
    pub fn validate<M, A: IntoAgent<M>>(&self, agent: A) -> Result<(), ResolveError> {
        agent.into_agent().validate(self)
    }
}

/// An agent's parameters could not be resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolveError {
    pub agent: Cow<'static, str>,
    /// Every unmet requirement, not just the first.
    pub missing: Vec<Requirement>,
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "agent `{}` cannot run; missing:", self.agent)?;
        for requirement in &self.missing {
            write!(f, "\n  - {requirement}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ResolveError {}
