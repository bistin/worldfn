use std::any::type_name;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

use crate::param::unmet;
use crate::{AgentParam, AgentWorld, ParamError, Requirement};

/// Parameter: an **already materialized** context snapshot of type `T`,
/// bound with [`AgentWorld::provide_context`].
///
/// M1 does not retrieve, rank, or budget anything: whoever builds the world
/// decides what the snapshot contains. Task-aware, per-invocation
/// materialization is M2 work (see `DESIGN.md`).
pub struct Context<T>(Arc<T>);

impl<T> Context<T> {
    pub fn new(snapshot: T) -> Self {
        Context(Arc::new(snapshot))
    }

    pub fn into_inner(self) -> Arc<T> {
        self.0
    }
}

impl<T> Clone for Context<T> {
    fn clone(&self) -> Self {
        Context(self.0.clone())
    }
}

impl<T> Deref for Context<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: fmt::Debug> fmt::Debug for Context<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Context").field(&self.0).finish()
    }
}

impl<T: Send + Sync + 'static> AgentParam for Context<T> {
    type State = Context<T>;

    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Context {
            type_name: type_name::<T>(),
        });
    }

    fn init(world: &AgentWorld) -> Result<Self::State, Vec<Requirement>> {
        world
            .resource::<Context<T>>()
            .cloned()
            .ok_or_else(unmet::<Self>)
    }

    fn resolve(state: &mut Self::State, _world: &AgentWorld) -> Result<Self, ParamError> {
        Ok(state.clone())
    }
}

/// Example snapshot type: memory entries somebody already judged relevant.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RelevantMemory {
    pub entries: Vec<String>,
}

impl RelevantMemory {
    pub fn new<I, S>(entries: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            entries: entries.into_iter().map(Into::into).collect(),
        }
    }
}
