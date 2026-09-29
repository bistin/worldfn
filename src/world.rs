use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::future::ready;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    Agent, AgentFuture, BindError, IntoAgent, Llm, LlmProvider, Memory, MemoryStore, RunError,
    Scope, SkillLibrary, Skills, Tool, ToolHandler, ToolSpec,
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
/// `Llm`, `Tool<T>` and `Memory` are stored as ordinary bindings keyed by
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

    /// Bind the store behind `Memory` (and so `Context<RelevantMemory<N>>`).
    pub fn provide_memory(&mut self, store: impl MemoryStore) -> Result<&mut Self, BindError> {
        self.provide(Memory::new(store))
    }

    /// Bind the skill library behind `Skills` (and the skill contexts).
    pub fn provide_skills(&mut self, library: SkillLibrary) -> Result<&mut Self, BindError> {
        self.provide(Skills::new(library))
    }

    /// Bind conversation storage behind `SessionLog` / `Conversation<N>`.
    pub fn provide_sessions(
        &mut self,
        store: impl crate::store::SessionStore,
    ) -> Result<&mut Self, BindError> {
        self.provide(crate::scoped::Sessions(Arc::new(store)))
    }

    /// Bind long-term storage behind `AccountMemory` / `Recall<N, Q>`.
    pub fn provide_account_memory(
        &mut self,
        store: impl crate::store::AccountMemoryStore,
    ) -> Result<&mut Self, BindError> {
        self.provide(crate::scoped::AccountMemories(Arc::new(store)))
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

    /// One-shot run with an empty invocation scope. See [`run_with`](Self::run_with).
    pub fn run<M, A: IntoAgent<M>>(&self, agent: A) -> AgentFuture<<A::Agent as Agent>::Output> {
        self.run_with(agent, Scope::new())
    }

    /// One-shot: prepare, then run with `scope` as this invocation's inputs.
    /// Pays for initialization every call; use [`prepare`](Self::prepare) to
    /// reuse it.
    ///
    /// Everything that reads the world happens *now*; the returned future is
    /// `Send + 'static`, borrows nothing, and can be spawned.
    pub fn run_with<M, A: IntoAgent<M>>(
        &self,
        agent: A,
        scope: Scope,
    ) -> AgentFuture<<A::Agent as Agent>::Output> {
        match self.prepare(agent) {
            Ok(mut agent) => agent.start(self, scope),
            Err(error) => Box::pin(ready(Err(error))),
        }
    }

    /// Run a prepared agent with an empty scope. See
    /// [`run_prepared_with`](Self::run_prepared_with).
    pub fn run_prepared<A: Agent + ?Sized>(&self, agent: &mut A) -> AgentFuture<A::Output> {
        agent.start(self, Scope::new())
    }

    /// Run an agent prepared against this world, reusing its parameter state.
    /// Fails with [`RunError::Stale`] or [`RunError::ForeignWorld`] rather than
    /// silently using outdated bindings.
    pub fn run_prepared_with<A: Agent + ?Sized>(
        &self,
        agent: &mut A,
        scope: Scope,
    ) -> AgentFuture<A::Output> {
        agent.start(self, scope)
    }
}
