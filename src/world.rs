use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    Agent, BindError, Context, IntoAgent, Llm, LlmProvider, RunError, Tool, ToolHandler, ToolSpec,
};

static NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(0);

/// Identifies a world and the version of its bindings. Prepared agents record
/// the stamp they were initialized against and refuse to run on any other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorldStamp {
    pub world: u64,
    pub generation: u64,
}

/// The container agents resolve their parameters from. Analogue of Bevy's
/// `World`, reduced to a type-keyed binding map with safe downcasts.
///
/// Each key has exactly one binding. [`provide`](Self::provide) refuses to
/// overwrite; [`replace`](Self::replace) rebinds explicitly and bumps the
/// binding generation, which invalidates every agent prepared earlier.
///
/// `Llm`, `Tool<T>` and `Context<T>` are stored as ordinary bindings keyed by
/// those logical types, so a fake is bound to the logical requirement
/// explicitly (`provide_llm(FakeLlm)` stores an `Llm`), never matched by the
/// fake's own `TypeId`.
pub struct AgentWorld {
    id: u64,
    generation: u64,
    bindings: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl std::fmt::Debug for AgentWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentWorld")
            .field("id", &self.id)
            .field("generation", &self.generation)
            .field("bindings", &self.bindings.len())
            .finish()
    }
}

impl Default for AgentWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentWorld {
    pub fn new() -> Self {
        Self {
            id: NEXT_WORLD_ID.fetch_add(1, Ordering::Relaxed),
            generation: 0,
            bindings: HashMap::new(),
        }
    }

    /// Bind a dependency, read by `Res<T>` parameters. Fails if `T` is
    /// already bound.
    pub fn provide<T: Send + Sync + 'static>(&mut self, value: T) -> Result<&mut Self, BindError> {
        if self.contains::<T>() {
            return Err(BindError {
                type_name: type_name::<T>(),
            });
        }
        Ok(self.replace(value))
    }

    /// Bind or rebind `T`. Invalidates every agent prepared against this world.
    pub fn replace<T: Send + Sync + 'static>(&mut self, value: T) -> &mut Self {
        self.bindings.insert(TypeId::of::<T>(), Arc::new(value));
        self.generation += 1;
        self
    }

    /// Bind the provider behind `Llm` parameters.
    pub fn provide_llm(&mut self, provider: impl LlmProvider) -> Result<&mut Self, BindError> {
        self.provide(Llm::new(provider))
    }

    /// Bind the handler behind `Tool<T>` parameters.
    pub fn provide_tool<T: ToolSpec>(
        &mut self,
        handler: impl ToolHandler<T>,
    ) -> Result<&mut Self, BindError> {
        self.provide(Tool::<T>::new(handler))
    }

    /// Bind an already materialized snapshot behind `Context<T>` parameters.
    pub fn provide_context<T: Send + Sync + 'static>(
        &mut self,
        snapshot: T,
    ) -> Result<&mut Self, BindError> {
        self.provide(Context::new(snapshot))
    }

    pub fn contains<T: Send + Sync + 'static>(&self) -> bool {
        self.bindings.contains_key(&TypeId::of::<T>())
    }

    pub fn resource<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.bindings.get(&TypeId::of::<T>())?.downcast_ref()
    }

    pub fn resource_arc<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.bindings
            .get(&TypeId::of::<T>())?
            .clone()
            .downcast()
            .ok()
    }

    pub(crate) fn stamp(&self) -> WorldStamp {
        WorldStamp {
            world: self.id,
            generation: self.generation,
        }
    }

    /// Convert and initialize an agent for repeated runs against this world.
    /// Parameter state is built once here and reused by every
    /// [`run_prepared`](Self::run_prepared).
    pub fn prepare<M, A: IntoAgent<M>>(&self, agent: A) -> Result<A::Agent, RunError> {
        let mut agent = agent.into_agent();
        agent.initialize(self)?;
        Ok(agent)
    }

    /// One-shot: prepare, resolve and run. Pays for initialization every call;
    /// use [`prepare`](Self::prepare) to reuse it.
    ///
    /// Everything that touches the world happens *now*; the returned future is
    /// `Send + 'static`, borrows nothing, and can be spawned.
    pub fn run<M, A: IntoAgent<M>>(
        &self,
        agent: A,
    ) -> impl Future<Output = Result<<A::Agent as Agent>::Output, RunError>> + Send + 'static + use<M, A>
    {
        let started = self.prepare(agent).and_then(|mut agent| agent.start(self));
        async move { Ok(started?.await) }
    }

    /// Run an agent prepared against this world, reusing its parameter state.
    /// Fails with [`RunError::Stale`] or [`RunError::ForeignWorld`] rather than
    /// silently using outdated bindings.
    pub fn run_prepared<A: Agent + ?Sized>(
        &self,
        agent: &mut A,
    ) -> impl Future<Output = Result<A::Output, RunError>> + Send + 'static + use<A> {
        let started = agent.start(self);
        async move { Ok(started?.await) }
    }
}
